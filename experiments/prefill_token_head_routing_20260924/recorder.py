"""Recorder using the generation bridge's single timing scope.
Static review revision; execution requires user approval.
"""
import json


def make_recorder(BaseRecorder, bridge_provider):
    class Recorder(BaseRecorder):
        def install(self):
            if self._installed:
                return
            bridge = bridge_provider()
            bridge.rows.clear()
            bridge.output.write_text('')
            # Use the bridge's one timing scope, not a second outer timer that
            # also counts JSON serialization and mistakes N=1 prefill for decode.
            def recorded_generate(*args, **kwargs):
                before = len(bridge.rows)
                result = self._original_generate(*args, **kwargs)
                if len(bridge.rows) != before + 1:
                    raise RuntimeError('expected exactly one generation telemetry row')
                detail = bridge.rows[-1]
                n = detail['valid_prompt_tokens']
                dense_pairs = self.model.config.num_hidden_layers * self.model.config.num_attention_heads * n * (n + 1) // 2
                selected = sum(layer['eligible_pairs'] for layer in detail['prefill_pairs_by_layer'])
                row = {'call_index':len(self.rows), 'method':self.method, **self.context,
                       'batch_size':1, 'input_tokens':detail['input_tokens'],
                       'input_ids_sha256':detail['input_ids_sha256'],
                       'generated_tokens':detail['generated_length'],
                       'prefill_latency_sec':detail['prefill_seconds'],
                       'decode_latency_sec':max(detail['total_seconds']-detail['prefill_seconds'],0.0),
                       'decode_forward_latency_sec':detail['decode_forward_seconds'],
                       'total_latency_sec':detail['total_seconds'],
                       'peak_memory_bytes':detail['peak_allocated_bytes'],
                       'selected_token_pairs':selected, 'causal_token_pairs':dense_pairs,
                       'global_token_keep_ratio':selected/dense_pairs,
                       'global_token_sparsity':1-selected/dense_pairs,
                       'stop_reason':detail['stop_reason'],
                       'pair_count_scope':'visible causal prefill pairs; not measured hardware operations',
                       'timing_scope':'bridge synchronized forward/wall clocks; excludes telemetry serialization'}
                self.rows.append(row)
                with self.output_path.open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                return result
            self.model.generate = recorded_generate
            self._installed = True

        def summary(self):
            result = super().summary()
            result['avg_decode_forward_latency_sec'] = (
                sum(row['decode_forward_latency_sec'] for row in self.rows)/len(self.rows)
                if self.rows else None)
            result['decode_latency_scope'] = 'legacy total-minus-prefill, includes dispatch/stopping; use decode_forward_latency_sec for model-only decode'
            return result
    return Recorder
