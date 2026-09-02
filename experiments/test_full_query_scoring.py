import math

import torch

from token_compacted_sparse import RepresentativeTokenFirstBlockSelector


def main() -> None:
    torch.manual_seed(7)
    batch_size, seq_len, num_groups, head_dim = 1, 11, 2, 3
    block_size = 4
    query = torch.randn(batch_size, seq_len, num_groups, head_dim)
    key = torch.randn_like(query)

    selector = object.__new__(RepresentativeTokenFirstBlockSelector)
    selector.query_score_mode = "full_query_mean"
    selector.probe_weights = (0.2, 0.3, 0.4, 0.1)
    actual, starts, ends, _ = selector._score_representative_keys(
        query, key, block_size=block_size
    )

    expected = torch.full_like(actual, float("-inf"))
    scale = math.sqrt(head_dim)
    for tile, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        for key_position in range(end):
            legal_queries = [
                query_position
                for query_position in range(start, end)
                if key_position <= query_position
            ]
            logits = torch.stack(
                [
                    torch.einsum(
                        "bhd,bhd->bh",
                        query[:, query_position],
                        key[:, key_position],
                    )
                    / scale
                    for query_position in legal_queries
                ]
            )
            expected[:, :, tile, key_position] = logits.mean(dim=0)

    causal = torch.arange(seq_len).view(1, 1, 1, -1) < ends.view(1, 1, -1, 1)
    torch.testing.assert_close(actual.masked_fill(~causal, 0), expected.masked_fill(~causal, 0))
    assert torch.isneginf(expected.masked_fill(causal, float("-inf"))).all()

    selector.query_score_mode = "four_probe_weighted"
    four_probe, _, _, lengths = selector._score_representative_keys(
        query, key, block_size=block_size
    )
    pooled = torch.stack(
        [query[:, start:end].mean(dim=1) for start, end in zip(starts, ends)],
        dim=1,
    )
    positions = starts[:, None] + torch.stack(
        ((lengths - 1) // 3, 2 * (lengths - 1) // 3, lengths - 1), dim=-1
    )
    manual_four_probe = (
        torch.einsum("bqhd,bthd->bhqt", pooled, key)
        * selector.probe_weights[3]
        / scale
    )
    for probe_index in range(3):
        probe_query = query.index_select(1, positions[:, probe_index])
        manual_four_probe += (
            torch.einsum("bqhd,bthd->bhqt", probe_query, key)
            * selector.probe_weights[probe_index]
            / scale
        )
    torch.testing.assert_close(four_probe, manual_four_probe)
    print("full_query_mean_matches_explicit_qk_average")
    print("four_probe_weighted_path_is_unchanged")


if __name__ == "__main__":
    main()
