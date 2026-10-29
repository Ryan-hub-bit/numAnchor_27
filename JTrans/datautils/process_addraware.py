"""
IDA-side richer extractor for future BinaryCorp/JTrans regeneration.

This script is intentionally separate from the original JTrans extractor.
It preserves more metadata so an address-aware corpus can be built later
without losing instruction-level location information.
"""

import os
import pickle
from collections import defaultdict

import idaapi
import idautils
import idc
import networkx as nx

from util.base import Binarybase


SAVEROOT = "./extract_addraware"
DATAROOT = "./dataset"


class BinaryDataAddressAware(Binarybase):
    def __init__(self, unstrip_path):
        super().__init__(unstrip_path)
        self.sections = self._collect_sections()
        self.binary_min_addr = min((sec["start"] for sec in self.sections), default=0)
        self.binary_max_addr = max((sec["end"] for sec in self.sections), default=0)

    def _collect_sections(self):
        sections = []
        for seg_ea in idautils.Segments():
            start = idc.get_segm_start(seg_ea)
            end = idc.get_segm_end(seg_ea)
            sections.append({
                "name": idc.get_segm_name(seg_ea),
                "start": start,
                "end": end,
            })
        return sections

    def _block_instructions(self, block):
        instructions = []
        curr_addr = block.start_ea
        func_addr_set = set(idautils.FuncItems(block.start_ea))
        while curr_addr < block.end_ea:
            if curr_addr not in func_addr_set and curr_addr != block.start_ea:
                break
            size = idc.get_item_size(curr_addr)
            raw = idc.get_bytes(curr_addr, size) or b""
            instructions.append({
                "ea": curr_addr,
                "disasm": idc.GetDisasm(curr_addr),
                "raw_hex": raw.hex(),
                "size": size,
            })
            next_addr = idc.next_head(curr_addr, block.end_ea)
            if next_addr <= curr_addr:
                break
            curr_addr = next_addr
        return instructions

    def get_cfg(self, func):
        nx_graph = nx.DiGraph()
        flowchart = idaapi.FlowChart(idaapi.get_func(func), flags=idaapi.FC_PREDS)
        for block in flowchart:
            instructions = self._block_instructions(block)
            if not instructions:
                continue
            nx_graph.add_node(
                block.start_ea,
                start_ea=block.start_ea,
                end_ea=block.end_ea,
                instructions=instructions,
            )
            for pred in block.preds():
                nx_graph.add_edge(pred.start_ea, block.start_ea)
            for succ in block.succs():
                nx_graph.add_edge(block.start_ea, succ.start_ea)
        return nx_graph

    def extract_all(self):
        for func in idautils.Functions():
            seg_name = idc.get_segm_name(func)
            if seg_name in [".plt", "extern", ".init", ".fini"]:
                continue
            func_obj = idaapi.get_func(func)
            if func_obj is None:
                continue
            yield {
                "name": self.addr2name[func],
                "function_addr": func,
                "function_start": func_obj.start_ea,
                "function_end": func_obj.end_ea,
                "cfg": self.get_cfg(func),
                "sections": self.sections,
                "binary_min_addr": self.binary_min_addr,
                "binary_max_addr": self.binary_max_addr,
            }


if __name__ == "__main__":
    assert os.path.exists(DATAROOT), "DATAROOT does not exist"
    os.makedirs(SAVEROOT, exist_ok=True)

    binary_abs_path = idc.get_input_file_path()
    filename = binary_abs_path.split("/")[-1][:-6]
    unstrip_path = os.path.join(DATAROOT, filename)
    idc.auto_wait()

    binary_data = BinaryDataAddressAware(unstrip_path)
    saved_dict = defaultdict(dict)
    saved_path = os.path.join(SAVEROOT, filename + "_extract_addraware.pkl")
    with open(saved_path, "wb") as f:
        for func_data in binary_data.extract_all():
            saved_dict[func_data["name"]] = func_data
        pickle.dump(dict(saved_dict), f)
    idc.qexit(0)
