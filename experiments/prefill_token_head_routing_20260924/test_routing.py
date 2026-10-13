"""Independent scalar-token oracle + actual Transformers Qwen3 integration."""
import copy
import json
import os
import unittest
from pathlib import Path
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM, LlamaConfig, LlamaForCausalLM, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from routing import routed_attention, repeat_project, RoutingController

torch.set_num_threads(2)
DEVICE = os.environ.get("ROUTING_TEST_DEVICE", "cpu")
FAMILY = os.environ.get("ROUTING_TEST_FAMILY", "llama")


def reference(q, k, v, positions, valid, mode, scale):
    # Deliberately no grouping, gather/scatter, production router or repetition.
    bsz, heads, n, dim = q.shape
    out = torch.zeros(bsz, n, dim, dtype=q.dtype, device=q.device)
    pairs = 0
    for b in range(bsz):
        for i in range(n):
            if not valid[b, i]:
                continue
            head = int(positions[b, i]) % heads
            kv = head // (heads // k.shape[1])
            keys = [j for j in range(n) if valid[b, j]
                    and positions[b, j] <= positions[b, i]
                    and (mode == 'q_only' or int(positions[b, j]) % heads == head)]
            scores = torch.stack([torch.dot(q[b, head, i].double(), k[b, kv, j].double()) * scale for j in keys])
            weights = scores.softmax(0)
            out[b, i] = sum(weights[t] * v[b, kv, j].double() for t, j in enumerate(keys))
            pairs += len(keys)
    return out, pairs


def rope_reference(x, positions, theta=10000.0):
    half = x.shape[-1] // 2
    freq = theta ** (-torch.arange(half, device=x.device).float() / half)
    angle = positions.float()[:, None, :, None] * freq
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * angle.cos() - b * angle.sin(), b * angle.cos() + a * angle.sin()), -1)


def small_model(kv=2):
    # Deliberately hidden_size != num_heads * head_dim; output bias tested too.
    config_class = LlamaConfig if FAMILY == "llama" else Qwen3Config
    model_class = LlamaForCausalLM if FAMILY == "llama" else Qwen3ForCausalLM
    cfg = config_class(hidden_size=24, intermediate_size=40, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=kv, head_dim=8,
                      vocab_size=43, max_position_embeddings=128, attention_bias=True,
                      rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
                      eos_token_id=42, pad_token_id=0)
    cfg._attn_implementation = os.environ.get("ROUTING_TEST_BACKEND", "eager")
    torch.manual_seed(192)
    return model_class(cfg).to(DEVICE).eval()


def prefill(model, ctl, ids, mask=None, positions=None):
    mask = torch.ones_like(ids) if mask is None else mask
    positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids) if positions is None else positions
    cache = DynamicCache(config=model.config)
    with torch.no_grad(), ctl.phase('prefill', mask.bool(), positions):
        out = model(ids, attention_mask=mask, position_ids=positions, past_key_values=cache, use_cache=True)
    return out


class TensorTests(unittest.TestCase):
    def test_reference_boundaries_mha_gqa_padding_rope(self):
        torch.manual_seed(77)
        for kv in (1, 2, 4):
            for n in (1, 3, 4, 7, 9):
                pos = torch.arange(n, device=DEVICE).expand(2, -1).clone()
                pos[1] += 5  # original, nonzero positions; never regroup-renumber
                valid = torch.ones(2, n, dtype=torch.bool, device=DEVICE)
                if n > 1:
                    valid[1, -2:] = False
                    valid[1, 0] = True
                q = torch.randn(2, 4, n, 8, device=DEVICE)
                k = torch.randn(2, kv, n, 8, device=DEVICE)
                v = torch.randn_like(k)
                qr, kr = rope_reference(q, pos), rope_reference(k, pos)
                for mode in ('q_only', 'q_kv'):
                    actual, counts = routed_attention(qr, kr, v, pos, valid, mode, 8 ** -0.5)
                    expected, pairs = reference(qr, kr, v, pos, valid, mode, 8 ** -0.5)
                    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)
                    self.assertEqual(counts['eligible_pairs'], pairs)
                    self.assertGreaterEqual(counts['matmul_pairs'], pairs)
                    self.assertTrue(torch.equal(actual[~valid], torch.zeros_like(actual[~valid])))
                    # Each original index is assigned exactly once; contiguous positions balanced.
                    groups = [(pos[0] % 4 == h).nonzero().flatten().tolist() for h in range(4)]
                    self.assertEqual(sorted(sum(groups, [])), list(range(n)))
                    self.assertLessEqual(max(map(len, groups)) - min(map(len, groups)), 1)

    def test_chunked_sdpa_reference(self):
        torch.manual_seed(43)
        q = torch.randn(2,4,19,8,device=DEVICE)
        k = torch.randn(2,2,19,8,device=DEVICE); v = torch.randn_like(k)
        pos = torch.arange(19,device=DEVICE).expand(2,-1)
        valid = torch.ones(2,19,device=DEVICE,dtype=torch.bool); valid[1,13:] = False
        for mode in ('q_only','q_kv'):
            expected, pairs = reference(q,k,v,pos,valid,mode,8**-0.5)
            for backend in ('eager','sdpa'):
                actual, counts = routed_attention(q,k,v,pos,valid,mode,8**-0.5,backend,query_chunk=2)
                torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
                self.assertEqual(pairs,counts['eligible_pairs'])

    def test_repeat_order_bias_and_sum(self):
        torch.manual_seed(12)
        m = torch.randn(2, 7, 8, device=DEVICE)
        proj = nn.Linear(32, 24, bias=True).to(DEVICE)
        expected = torch.stack([torch.cat([row, row, row, row]) for row in m.reshape(-1, 8)]).reshape(2, 7, 32)
        torch.testing.assert_close(repeat_project(m, proj, 4), proj(expected), rtol=0, atol=0)
        torch.testing.assert_close(repeat_project(m, proj, 4, True), proj(expected), rtol=3e-5, atol=2e-6)
        # Selected-head block is explicitly a different method.
        selected = torch.nn.functional.linear(m, proj.weight[:, :8], proj.bias)
        self.assertFalse(torch.allclose(selected, proj(expected)))

    def test_rope_against_library(self):
        q = torch.randn(2, 4, 7, 8, device=DEVICE)
        k = torch.randn(2, 2, 7, 8, device=DEVICE)
        pos = torch.arange(7, device=DEVICE).expand(2, -1) + 11
        angle = pos[..., None] * (10000.0 ** (-torch.arange(4, device=DEVICE).float() / 4))
        angle = torch.cat([angle, angle], -1)
        a, b = apply_rotary_pos_emb(q, k, angle.cos(), angle.sin())
        torch.testing.assert_close(a, rope_reference(q, pos))
        torch.testing.assert_close(b, rope_reference(k, pos))


class ModelTests(unittest.TestCase):
    def test_prefill_cache_provenance_dense_decode_and_future(self):
        for mode in ('dense', 'q_only', 'q_kv'):
            model = small_model()
            pristine = copy.deepcopy(model)
            ctl = RoutingController(model, mode)
            captured = {}
            hooks = []
            for li, layer in enumerate(model.model.layers):
                def capture(module, args, index=li):
                    captured[index] = args[0].detach().clone()
                hooks.append(layer.self_attn.k_proj.register_forward_pre_hook(capture))
            ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8]], device=DEVICE)
            out = prefill(model, ctl, ids)
            for hook in hooks:
                hook.remove()
            # Cache must match projections of this method's actual per-layer inputs.
            for li, layer in enumerate(model.model.layers):
                attn = layer.self_attn
                with torch.no_grad():
                    rawk = attn.k_proj(captured[li]).view(1, 7, 2, 8)
                    if hasattr(attn, "k_norm"):
                        rawk = attn.k_norm(rawk)
                    rawk = rawk.transpose(1, 2)
                    rawv = attn.v_proj(captured[li]).view(1, 7, 2, 8).transpose(1, 2)
                expectedk = rope_reference(rawk, torch.arange(7, device=DEVICE)[None])
                torch.testing.assert_close(out.past_key_values.layers[li].keys, expectedk)
                torch.testing.assert_close(out.past_key_values.layers[li].values, rawv)
                self.assertEqual(tuple(expectedk.shape), (1, 2, 7, 8))
            changed = ids.clone(); changed[0, 4:] = torch.tensor([20, 21, 22], device=DEVICE)
            other = prefill(model, ctl, changed)
            torch.testing.assert_close(out.logits[:, :4], other.logits[:, :4], rtol=0, atol=0)
            # Unpatched stock dense model with a copy of ROUTED cache, not a dense prefill cache.
            cache = out.past_key_values
            reference_cache = copy.deepcopy(cache)
            for step in range(4):
                token = torch.tensor([[9 + step]], device=DEVICE)
                mask = torch.ones(1, 8 + step, dtype=torch.long, device=DEVICE)
                pos = torch.tensor([[7 + step]], device=DEVICE)
                with torch.no_grad(), ctl.phase('decode'):
                    got = model(token, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)
                with torch.no_grad():
                    ref = pristine(token, attention_mask=mask, position_ids=pos, past_key_values=reference_cache, use_cache=True)
                torch.testing.assert_close(got.logits, ref.logits, rtol=0, atol=0)
                for a, b in zip(cache.layers, reference_cache.layers):
                    torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
                    self.assertEqual(a.keys.shape[-2], 8 + step)
            # Confirm routed layer 2 states/cache differ from a separate dense prefill.
            densectl = RoutingController(pristine, 'dense')
            dense = prefill(pristine, densectl, ids)
            if mode == 'dense':
                torch.testing.assert_close(out.logits, dense.logits, rtol=0, atol=0)
            else:
                self.assertFalse(torch.allclose(out.logits, dense.logits))
                self.assertFalse(torch.allclose(out.past_key_values.layers[1].keys[..., :7, :], dense.past_key_values.layers[1].keys))
            densectl.close(); ctl.close()

    def test_padded_variable_batch_matches_unpadded(self):
        for mode in ('dense', 'q_only', 'q_kv'):
            model = small_model(); ctl = RoutingController(model, mode)
            for left in (False, True):
                ids = torch.tensor([[2,3,4,5,6,7,8], [0,0,0,0,9,10,11] if left else [9,10,11,0,0,0,0]], device=DEVICE)
                mask = (ids != 0).long()
                pos = (mask.cumsum(-1) - 1).clamp(min=0)
                batch = prefill(model, ctl, ids, mask, pos)
                for b in range(2):
                    keep = mask[b].bool()
                    solo = prefill(model, ctl, ids[b:b+1, keep])
                    torch.testing.assert_close(batch.logits[b, keep], solo.logits[0], atol=2e-6, rtol=2e-5)
                # Left/right-padded prefill -> standard dense decode, original logical positions.
                token = torch.tensor([[12],[13]], device=DEVICE)
                fullmask = torch.cat([mask, torch.ones(2,1,device=DEVICE,dtype=mask.dtype)], 1)
                with torch.no_grad(), ctl.phase('decode'):
                    decoded = model(token, attention_mask=fullmask, position_ids=mask.sum(-1,keepdim=True), past_key_values=batch.past_key_values, use_cache=True)
                for b in range(2):
                    keep = mask[b].bool(); solo = prefill(model, ctl, ids[b:b+1,keep])
                    with torch.no_grad(), ctl.phase('decode'):
                        ref = model(token[b:b+1], attention_mask=torch.ones(1,int(keep.sum())+1,device=DEVICE,dtype=mask.dtype), position_ids=mask[b:b+1].sum(-1,keepdim=True), past_key_values=solo.past_key_values, use_cache=True)
                    torch.testing.assert_close(decoded.logits[b], ref.logits[0], atol=2e-6, rtol=2e-5)
            ctl.close()

    def test_hf_generation_bridge(self):
        from generation_bridge import GenerationBridge
        ids = torch.tensor([[2,3,4,5,6,7,8]],device=DEVICE)
        for mode in ('dense','q_only','q_kv'):
            model = small_model()
            ctl = RoutingController(model,mode)
            out = prefill(model,ctl,ids)
            expected = []
            mask = torch.ones_like(ids)
            for step in range(5):
                token = out.logits[:,-1].argmax(-1,keepdim=True)
                expected.append(int(token[0,0]))
                if expected[-1] == 42 or step == 4:
                    break
                mask = torch.cat([mask,torch.ones_like(token)],-1)
                with torch.no_grad(),ctl.phase('decode'):
                    out = model(token,attention_mask=mask,position_ids=torch.full_like(token,ids.shape[1]+step),past_key_values=out.past_key_values,use_cache=True)
            ctl.close()
            bridge = GenerationBridge(model,mode)
            result = model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=5,do_sample=False,pad_token_id=0,eos_token_id=42)
            self.assertEqual(result[0,ids.shape[1]:].tolist(),expected)
            self.assertEqual(bridge.rows[0]['generated_token_ids'],expected)
            self.assertEqual(len(bridge.rows[0]['prefill_pairs_by_layer']),2)
            # Immediate EOS retains native HF stopping semantics.
            immediate = model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=5,do_sample=False,pad_token_id=0,eos_token_id=expected[0])
            self.assertEqual(immediate.shape[1]-ids.shape[1],1)
            self.assertEqual(bridge.rows[-1]['stop_reason'],'eos')
            from transformers import StoppingCriteria, StoppingCriteriaList
            class StopAfterTwo(StoppingCriteria):
                def __call__(self,input_ids,scores,**kwargs):
                    return torch.full((input_ids.shape[0],),input_ids.shape[1]>=ids.shape[1]+2,device=input_ids.device,dtype=torch.bool)
            stopped = model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=5,do_sample=False,pad_token_id=0,eos_token_id=[],stopping_criteria=StoppingCriteriaList([StopAfterTwo()]))
            self.assertEqual(stopped.shape[1]-ids.shape[1],2)
            self.assertEqual(bridge.rows[-1]['stop_reason'],'stopping_criteria')
            bridge.close()

    def test_one_token_prompt_and_rejections(self):
        model = small_model(); ctl = RoutingController(model, 'q_kv')
        ids = torch.tensor([[5]], device=DEVICE)
        out = prefill(model, ctl, ids)
        self.assertEqual(len(ctl.counts), 2)
        self.assertEqual(ctl.counts[0]['eligible_pairs'], 1)
        with self.assertRaisesRegex(ValueError, 'chunked'), torch.no_grad(), ctl.phase('prefill', torch.ones_like(ids), torch.zeros_like(ids)):
            model(ids, attention_mask=torch.ones_like(ids), position_ids=torch.zeros_like(ids), past_key_values=out.past_key_values)
        with self.assertRaisesRegex(ValueError, 'exactly one'), torch.no_grad(), ctl.phase('decode'):
            model(ids.expand(1,2), past_key_values=out.past_key_values)
        with self.assertRaisesRegex(RuntimeError, 'explicit phase'), torch.no_grad():
            model(ids)
        with self.assertRaisesRegex(ValueError, 'metadata'), torch.no_grad(), ctl.phase('prefill', torch.ones_like(ids), torch.ones_like(ids)):
            model(ids, attention_mask=torch.ones_like(ids), position_ids=torch.zeros_like(ids))
        ctl.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
