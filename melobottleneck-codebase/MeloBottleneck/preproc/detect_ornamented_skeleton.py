from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mido
from miditoolkit import MidiFile


class ChannelResolutionError(RuntimeError):
    """Raised when Ornamented / Skeleton cannot be resolved unambiguously."""


@dataclass
class CandidateChannel:
    track_index: int
    channel: int
    program: int
    track_name: str
    instrument_name: str
    note_count: int

    @property
    def debug_name(self) -> str:
        tn = self.track_name or ""
        ins = self.instrument_name or ""
        return f"track_name={tn!r}, instrument_name={ins!r}"

    def to_dict(self) -> dict:
        return {
            "track_index": self.track_index,
            "channel": self.channel,
            "program": self.program,
            "track_name": self.track_name,
            "instrument_name": self.instrument_name,
            "note_count": self.note_count,
        }


@dataclass
class ResolutionResult:
    strategy: str  # "name" or "note_count"
    ornamented: CandidateChannel
    skeleton: CandidateChannel

    def note_count_dict(self) -> Dict[str, int]:
        return {
            "Ornamented": self.ornamented.note_count,
            "Skeleton": self.skeleton.note_count,
        }


def _norm_name(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())


def _label_match_score(candidate: CandidateChannel, label: str) -> int:
    """
    返回匹配分数：
    3 = 规范化后完全相等
    2 = 原始字符串中以单词/分隔形式出现
    1 = 规范化后包含
    0 = 不匹配
    """
    target = _norm_name(label)
    if not target:
        return 0

    raw_names = [candidate.track_name or "", candidate.instrument_name or ""]
    norm_names = [_norm_name(x) for x in raw_names if x]

    # 最高优先：规范化后完全相等
    if any(n == target for n in norm_names):
        return 3

    # 中优先：原始文本按单词匹配（大小写不敏感）
    word_pat = re.compile(rf"(?i)(?:^|[\s_\-]){re.escape(label)}(?:$|[\s_\-])")
    if any(word_pat.search(x) for x in raw_names if x):
        return 2

    # 低优先：规范化后包含
    if any(target in n for n in norm_names):
        return 1

    return 0


def _pick_unique_by_label(
    candidates: List[CandidateChannel], label: str
) -> Optional[CandidateChannel]:
    scored = [(c, _label_match_score(c, label)) for c in candidates]
    scored = [(c, s) for c, s in scored if s > 0]
    if not scored:
        return None

    best_score = max(s for _, s in scored)
    best = [c for c, s in scored if s == best_score]

    if len(best) == 1:
        return best[0]
    return None


def _collect_sounding_candidates_with_mido(midi_path: str | Path) -> List[CandidateChannel]:
    """
    以 (track_index, channel, current_program) 为候选粒度统计：
    - track_name
    - instrument_name
    - 实际发声音符数（note_on velocity > 0）
    """
    mf = mido.MidiFile(str(midi_path))

    candidates: Dict[Tuple[int, int, int], CandidateChannel] = {}

    for track_index, track in enumerate(mf.tracks):
        # 先抽这一整条 track 的 meta name
        track_name = ""
        instrument_name = ""

        for msg in track:
            if msg.is_meta and msg.type == "track_name" and getattr(msg, "name", ""):
                if not track_name:
                    track_name = msg.name
            elif msg.is_meta and msg.type == "instrument_name" and getattr(msg, "name", ""):
                if not instrument_name:
                    instrument_name = msg.name

        # 再按事件顺序统计当前 program 下的 note_on 数
        current_program = [0] * 16  # MIDI 默认 program 0

        for msg in track:
            if msg.is_meta:
                continue

            if msg.type == "program_change":
                current_program[msg.channel] = msg.program
                continue

            if msg.type == "note_on" and msg.velocity > 0:
                key = (track_index, msg.channel, current_program[msg.channel])
                if key not in candidates:
                    candidates[key] = CandidateChannel(
                        track_index=track_index,
                        channel=msg.channel,
                        program=current_program[msg.channel],
                        track_name=track_name,
                        instrument_name=instrument_name,
                        note_count=0,
                    )
                candidates[key].note_count += 1

    # 只保留真正有发声音符的候选
    out = [c for c in candidates.values() if c.note_count > 0]
    out.sort(key=lambda x: (x.track_index, x.channel, x.program))
    return out


def _load_with_miditoolkit(midi_path: str | Path) -> MidiFile:
    """
    主读取接口。当前版本主要用于解析有效性检查，并为后续扩展保留入口。
    """
    midi_obj = MidiFile(str(midi_path))
    return midi_obj


def resolve_ornamented_skeleton(midi_path: str | Path) -> ResolutionResult:
    """
    两阶段判别：
    1. 先按 Ornamented / Skeleton 名字判别；
    2. 失败则按 note_count 判别（要求恰好两个有声候选，且计数不相等）。

    这里默认沿用你的业务语义：
    - 音符数更多的一路视为 Ornamented
    - 音符数更少的一路视为 Skeleton
    """
    midi_obj = _load_with_miditoolkit(midi_path)
    candidates = _collect_sounding_candidates_with_mido(midi_path)

    if not midi_obj.instruments:
        raise ChannelResolutionError(
            "miditoolkit 未读取到任何 instrument；这不是一个正常的可解析 MIDI，或文件里没有可用音符数据。"
        )

    if not candidates:
        raise ChannelResolutionError("没有找到任何带实际发声音符的候选通道。")

    # 方案 1：按名字判别
    ornamented_by_name = _pick_unique_by_label(candidates, "Ornamented")
    skeleton_by_name = _pick_unique_by_label(candidates, "Skeleton")

    if (
        ornamented_by_name is not None
        and skeleton_by_name is not None
        and ornamented_by_name is not skeleton_by_name
    ):
        return ResolutionResult(
            strategy="name",
            ornamented=ornamented_by_name,
            skeleton=skeleton_by_name,
        )

    # 方案 2：按 note_count 回退
    if len(candidates) != 2:
        debug_lines = "\n".join(f"  - {c.to_dict()}" for c in candidates)
        raise ChannelResolutionError(
            "名字判别失败，且 note_count 回退要求“恰好两个带发声音符的候选通道”未满足。\n"
            f"当前候选数 = {len(candidates)}\n"
            f"{debug_lines}"
        )

    a, b = candidates
    if a.note_count == b.note_count:
        raise ChannelResolutionError(
            "名字判别失败，且两个候选通道的实际发声音符数相同，无法按 note_count 判别。\n"
            f"candidate A: {a.to_dict()}\n"
            f"candidate B: {b.to_dict()}"
        )

    if a.note_count > b.note_count:
        ornamented, skeleton = a, b
    else:
        ornamented, skeleton = b, a

    return ResolutionResult(
        strategy="note_count",
        ornamented=ornamented,
        skeleton=skeleton,
    )


def get_ornamented_skeleton_note_counts(midi_path: str | Path) -> Dict[str, int]:
    """
    当前对外主接口：只返回 Ornamented / Skeleton 的音符数。
    后面如果你要扩展成返回 channel / program / 名字等，可以直接复用 resolve_ornamented_skeleton。
    """
    result = resolve_ornamented_skeleton(midi_path)
    return result.note_count_dict()


def main():
    parser = argparse.ArgumentParser(
        description="Resolve Ornamented / Skeleton from an FL Studio-exported MIDI."
    )
    parser.add_argument("midi_path", type=str, help="Path to the MIDI file")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra resolved channel metadata",
    )
    args = parser.parse_args()

    result = resolve_ornamented_skeleton(args.midi_path)

    print(f"Ornamented notes: {result.ornamented.note_count}")
    print(f"Skeleton notes: {result.skeleton.note_count}")

    if args.verbose:
        print(f"\nResolution strategy: {result.strategy}")
        print("\n[Ornamented]")
        print(result.ornamented.to_dict())
        print("\n[Skeleton]")
        print(result.skeleton.to_dict())


if __name__ == "__main__":
    main()

# usage:
# python detect_ornamented_skeleton.py "J:\ACADEMIC\GRADPROJ\jiugongdacheng_skeleton\test_set\1\奉时春 月令承应.mid" --verbose
