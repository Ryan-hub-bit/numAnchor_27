import os
import sys
import tempfile
import unittest

import torch


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TEST_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from address_embedding import AddressAwareBERTEmbedding
from create_vocab import preprocess_line
from dataloader import InstructionMaskingDataset
from model import AddressAwareBERT
from probing.finetune_pair import UnitSequenceBuilder
from probing.probe_eval import (
    build_call_target_pairs,
    build_dref_retrieval_queries,
    build_jump_target_pairs,
    build_same_bb_pairs,
    build_same_dref_pairs,
    reconstruct_blocks,
)
from vocab import WordVocab
from data_generator.section_utils import (
    SECTION_KIND_BSS,
    SECTION_KIND_DATA,
    SECTION_KIND_OTHER,
    SECTION_KIND_RODATA,
    classify_data_section,
    daddr_token_for_kind,
    default_daddr_operand,
    format_daddr_operand,
)


def build_vocab():
    token_lists = [[
        "mov", "add", "jmp", "sub", "call", "xor", "rax", "rbx", "rcx", "rdx",
        "address", "daddr_rodata", "daddr_bss", "daddr_data", "daddr_other",
        "var", "[", "]", "+", "-", "*", ",",
    ]]
    return WordVocab(token_lists, min_freq=1)


class AddressAwareSmokeTests(unittest.TestCase):
    def test_section_kind_classification_and_formatting(self):
        self.assertEqual(classify_data_section(".rodata.str1.1"), (True, SECTION_KIND_RODATA))
        self.assertEqual(classify_data_section(".bss"), (True, SECTION_KIND_BSS))
        self.assertEqual(classify_data_section(".data"), (True, SECTION_KIND_DATA))
        self.assertEqual(classify_data_section(".got.plt"), (True, SECTION_KIND_OTHER))
        self.assertEqual(classify_data_section(".text"), (False, SECTION_KIND_OTHER))
        self.assertEqual(daddr_token_for_kind(SECTION_KIND_RODATA), "daddr_rodata")
        self.assertEqual(
            format_daddr_operand("0x404000", 0.25, 0.5, SECTION_KIND_OTHER),
            "daddr_other(0x404000:0.25000000:0.50000000)",
        )
        self.assertEqual(default_daddr_operand("0x404000"), "daddr_other(0x404000:0.00000000:0.00000000)")

    def test_vocab_preprocess_canonicalizes_legacy_daddr(self):
        tokens = preprocess_line(
            "mov(0x401000:0.1:0.2:0.3) daddr(0x404000:0.7:0.2:1) daddr_bss(0x405000:0.8:0.4)"
        )
        self.assertEqual(tokens, ["mov", "daddr_rodata", "daddr_bss"])

    def test_dataloader_parser_reads_code_binary_pos_and_daddr_kind(self):
        vocab = build_vocab()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as cfg:
            cfg.write("mov(0x401000:0.1:0.2:0.3) rax\n")
            cfg_path = cfg.name
        try:
            dataset = InstructionMaskingDataset(
                cfg_corpus_path=cfg_path,
                dfg_corpus_path=None,
                vocab=vocab,
                seq_len=32,
                token_mask_prob=0.0,
                instruction_mask_prob=0.0,
                enable_imd=False,
            )
            parsed = dataset._parse_instruction(
                "mov(0x401000:0.12500000:0.30000000:0.50000000) "
                "address(0x401010:0.12500000:0.40000000:0.00000000) "
                "daddr_rodata(0x404000:0.70000000:0.20000000) "
                "daddr(0x405000:0.75000000:0.30000000:2) var(0x10)"
            )
            tokens, positions, var_offsets, section_kind_ids, addr_type_ids = parsed

            self.assertEqual(tokens[0], "mov")
            self.assertAlmostEqual(positions[0][0], 0.125)
            self.assertEqual(addr_type_ids[0], 1)

            addr_idx = tokens.index("address")
            self.assertAlmostEqual(positions[addr_idx][0], 0.125)
            self.assertEqual(addr_type_ids[addr_idx], 1)

            daddr_idx = tokens.index("daddr_rodata")
            self.assertAlmostEqual(positions[daddr_idx][0], 0.7)
            self.assertAlmostEqual(positions[daddr_idx][1], 0.2)
            self.assertEqual(section_kind_ids[daddr_idx], 1)
            self.assertEqual(addr_type_ids[daddr_idx], 2)

            legacy_daddr_idx = tokens.index("daddr_bss")
            self.assertAlmostEqual(positions[legacy_daddr_idx][0], 0.75)
            self.assertAlmostEqual(positions[legacy_daddr_idx][1], 0.3)
            self.assertEqual(section_kind_ids[legacy_daddr_idx], 2)
            self.assertEqual(addr_type_ids[legacy_daddr_idx], 2)

            var_idx = tokens.index("var")
            self.assertEqual(var_offsets[var_idx], 16)
            self.assertEqual(addr_type_ids[var_idx], 3)
        finally:
            os.unlink(cfg_path)

    def test_embedding_uses_code_binary_pos_and_daddr_section_token(self):
        vocab = build_vocab()
        mov_id = vocab.stoi["mov"]
        daddr_rodata_id = vocab.stoi["daddr_rodata"]
        daddr_other_id = vocab.stoi["daddr_other"]

        embedding = AddressAwareBERTEmbedding(
            vocab_size=len(vocab),
            embed_size=24,
            dropout=0.0,
            max_len=8,
            use_address_embedding=True,
            use_var_embedding=False,
        )
        embedding.set_special_token_ids([daddr_rodata_id, daddr_other_id])
        embedding.eval()

        segment = torch.tensor([[1]], dtype=torch.long)

        code_low = embedding(
            torch.tensor([[mov_id]], dtype=torch.long),
            segment,
            torch.tensor([[0.1]], dtype=torch.float),
            torch.tensor([[0.2]], dtype=torch.float),
            torch.tensor([[0.3]], dtype=torch.float),
            var_offsets=None,
            section_kind_ids=torch.tensor([[0]], dtype=torch.long),
            addr_type_ids=torch.tensor([[1]], dtype=torch.long),
        )
        code_high = embedding(
            torch.tensor([[mov_id]], dtype=torch.long),
            segment,
            torch.tensor([[0.9]], dtype=torch.float),
            torch.tensor([[0.2]], dtype=torch.float),
            torch.tensor([[0.3]], dtype=torch.float),
            var_offsets=None,
            section_kind_ids=torch.tensor([[0]], dtype=torch.long),
            addr_type_ids=torch.tensor([[1]], dtype=torch.long),
        )
        self.assertFalse(torch.allclose(code_low, code_high))

        daddr_rodata = embedding(
            torch.tensor([[daddr_rodata_id]], dtype=torch.long),
            segment,
            torch.tensor([[0.7]], dtype=torch.float),
            torch.tensor([[0.2]], dtype=torch.float),
            torch.tensor([[-1.0]], dtype=torch.float),
            var_offsets=None,
            section_kind_ids=torch.tensor([[1]], dtype=torch.long),
            addr_type_ids=torch.tensor([[2]], dtype=torch.long),
        )
        daddr_other = embedding(
            torch.tensor([[daddr_other_id]], dtype=torch.long),
            segment,
            torch.tensor([[0.7]], dtype=torch.float),
            torch.tensor([[0.2]], dtype=torch.float),
            torch.tensor([[-1.0]], dtype=torch.float),
            var_offsets=None,
            section_kind_ids=torch.tensor([[4]], dtype=torch.long),
            addr_type_ids=torch.tensor([[2]], dtype=torch.long),
        )
        self.assertFalse(torch.allclose(daddr_rodata, daddr_other))

    def test_probing_pair_builders_and_forward_smoke(self):
        vocab = build_vocab()
        function_blocks = {
            "f1": [
                "mov(0x401000:0.10000000:0.20000000:0.00000000) rax daddr_rodata(0x404000:0.70000000:0.20000000)",
                "add(0x401004:0.10000000:0.20000000:0.50000000) rbx rcx",
                "jmp(0x401008:0.10000000:0.20000000:1.00000000) address(0x401010:0.10000000:0.40000000:0.00000000)",
                "sub(0x401010:0.10000000:0.40000000:0.00000000) rax var(0x10)",
            ],
        }

        instructions_by_func, blocks_by_func = reconstruct_blocks(function_blocks)
        same_bb_pairs, same_bb_units = build_same_bb_pairs(blocks_by_func, ["f1"], max_pairs=8, seed=7)
        jump_pairs, jump_units = build_jump_target_pairs(
            instructions_by_func, ["f1"], max_pairs=8, seed=11, target_opcode=""
        )

        self.assertTrue(same_bb_pairs)
        self.assertTrue(jump_pairs)

        builder = UnitSequenceBuilder(vocab=vocab, seq_len=32, max_instructions=4)
        encoded = builder.build(same_bb_units[same_bb_pairs[0].left_id].instructions)

        model = AddressAwareBERT(
            vocab_size=len(vocab),
            hidden=32,
            n_layers=2,
            attn_heads=4,
            dropout=0.0,
            max_len=32,
            use_address_embedding=True,
            use_var_embedding=True,
        )
        model.embedding.set_special_token_ids(
            [vocab.stoi["daddr_rodata"], vocab.stoi["daddr_bss"], vocab.stoi["daddr_data"], vocab.stoi["daddr_other"]]
        )
        model.eval()

        hidden = model(
            encoded["input_ids"].unsqueeze(0),
            encoded["segment_labels"].unsqueeze(0),
            encoded["binary_pos"].unsqueeze(0),
            encoded["function_pos"].unsqueeze(0),
            encoded["bb_pos"].unsqueeze(0),
            encoded["var_offsets"].unsqueeze(0),
            section_kind_ids=encoded["section_kind_ids"].unsqueeze(0),
            addr_type_ids=encoded["addr_type_ids"].unsqueeze(0),
        )
        self.assertEqual(tuple(hidden.shape), (1, 32, 32))

    def test_call_and_dref_probe_builders(self):
        function_blocks = {
            "f1": [
                "mov(0x1000:0.10000000:0.10000000:0.00000000) rax daddr_rodata(0x5000:0.70000000:0.10000000)",
                "call(0x1004:0.10000000:0.10000000:0.50000000) address(0x2000:0.30000000:0.00000000:0.00000000)",
            ],
            "f2": [
                "sub(0x2000:0.30000000:0.00000000:0.00000000) rbx rcx",
                "mov(0x2004:0.30000000:0.00000000:0.50000000) rcx daddr_rodata(0x5000:0.70000000:0.10000000)",
            ],
            "f3": [
                "xor(0x3000:0.60000000:0.00000000:0.00000000) rdx rdx",
                "mov(0x3004:0.60000000:0.00000000:0.50000000) rdx daddr_bss(0x6000:0.90000000:0.20000000)",
            ],
        }
        binary_by_func = {"f1": "binA", "f2": "binA", "f3": "binA"}

        instructions_by_func, blocks_by_func = reconstruct_blocks(function_blocks)
        call_pairs, call_units = build_call_target_pairs(
            instructions_by_func,
            blocks_by_func,
            ["f1", "f2", "f3"],
            binary_by_func,
            max_pairs=8,
            seed=13,
        )
        dref_pairs, dref_units = build_same_dref_pairs(
            instructions_by_func,
            ["f1", "f2", "f3"],
            binary_by_func,
            max_pairs=8,
            seed=17,
        )
        retrieval_queries, retrieval_units = build_dref_retrieval_queries(
            instructions_by_func,
            ["f1", "f2", "f3"],
            binary_by_func,
            max_queries=4,
            pool_size=2,
            seed=19,
        )

        self.assertTrue(call_pairs)
        self.assertTrue(any(pair.label == 1 for pair in call_pairs))
        self.assertTrue(dref_pairs)
        self.assertTrue(any(pair.label == 1 for pair in dref_pairs))
        self.assertTrue(retrieval_queries)
        self.assertEqual(len(retrieval_queries[0].candidate_ids), 2)
        self.assertIn(retrieval_queries[0].positive_id, retrieval_queries[0].candidate_ids)
        self.assertTrue(call_units)
        self.assertTrue(dref_units)
        self.assertTrue(retrieval_units)


if __name__ == "__main__":
    unittest.main()
