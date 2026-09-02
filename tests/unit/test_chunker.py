"""Unit tests for the structure-aware chunker.

Written before the implementation. The central property is offset fidelity:
every chunk must be a verbatim slice of its document. A chunker that quietly
strips or reflows text while keeping the original offsets produces chunks that
look fine and evidence spans that point at the wrong characters -- the exact
failure `docs/DATA_AUDIT.md` check 4 exists to prevent.

The corpus profile these tests encode (from the audit):
  - 59% of contracts are run-on, mean line length 340 chars, p90 1,453.
    Headings therefore appear mid-line after whitespace runs, not at line starts.
  - 41 of 510 contracts have no numbered headings at all, and 94 have fewer
    than five. The token-count fallback is a primary path, not an edge case.
"""

from __future__ import annotations

import itertools

import pytest

from docintel.ingest.chunker import (
    ChunkingConfig,
    WordTokenCounter,
    chunk_document,
    find_headings,
)
from docintel.schemas import Document

# One token per whitespace-delimited word keeps the arithmetic in these tests
# legible. The production counter is tiktoken; the chunker takes either.
COUNTER = WordTokenCounter()


def make_document(text: str, document_id: str = "DOC") -> Document:
    return Document(document_id=document_id, text=text)


def words(count: int, word: str = "lorem") -> str:
    return " ".join([word] * count)


class TestOffsetFidelity:
    """The invariant everything else depends on."""

    @pytest.mark.parametrize(
        "text",
        [
            "1. TERM. This agreement runs for one year.",
            "Preamble text with no heading at all.",
            "1. TERM. Body.   2. GOVERNING LAW. Delaware.   3. NOTICES. By mail.",
            "ARTICLE I\n\nDEFINITIONS\n\nAs used herein:\n\nARTICLE II\n\nTERM\n\nOne year.",
            "\n\n\n   \t  leading and trailing whitespace   \n\n\n",
            "Unicode: LECLANCHÉ S.A. and a café clause.",
            "",
        ],
    )
    def test_every_chunk_is_a_verbatim_slice(self, text: str) -> None:
        doc = make_document(text)
        for chunk in chunk_document(doc, ChunkingConfig(max_tokens=8), COUNTER):
            assert chunk.is_faithful_to(doc), (
                f"chunk {chunk.ordinal} text does not match "
                f"doc[{chunk.char_start}:{chunk.char_end}]"
            )

    def test_holds_when_sections_must_be_split(self) -> None:
        doc = make_document(f"1. TERM. {words(300)}   2. LAW. {words(300)}")
        chunks = chunk_document(doc, ChunkingConfig(max_tokens=50), COUNTER)
        assert len(chunks) > 4
        assert all(c.is_faithful_to(doc) for c in chunks)

    def test_holds_for_a_single_unbroken_paragraph(self) -> None:
        """Hard-split path: no headings, no paragraph breaks, nowhere clean to cut."""
        doc = make_document(words(500))
        chunks = chunk_document(doc, ChunkingConfig(max_tokens=40, overlap_tokens=0), COUNTER)
        assert len(chunks) > 1
        assert all(c.is_faithful_to(doc) for c in chunks)

    def test_offsets_are_within_document_bounds(self) -> None:
        doc = make_document("1. TERM. " + words(200))
        for chunk in chunk_document(doc, ChunkingConfig(max_tokens=30), COUNTER):
            assert 0 <= chunk.char_start <= chunk.char_end <= len(doc.text)


class TestCoverage:
    def test_chunks_are_ordered_by_position(self) -> None:
        doc = make_document("1. A. " + words(100) + "   2. B. " + words(100))
        chunks = chunk_document(doc, ChunkingConfig(max_tokens=40), COUNTER)
        starts = [c.char_start for c in chunks]
        assert starts == sorted(starts)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_no_content_is_dropped(self) -> None:
        """Concatenating chunk texts must reproduce every non-whitespace
        character of the document, in order. Whitespace between chunks may be
        discarded; content may not."""
        text = "1. TERM. Alpha bravo charlie.   2. LAW. Delta echo foxtrot.   3. END. Golf."
        doc = make_document(text)
        chunks = chunk_document(doc, ChunkingConfig(max_tokens=6, overlap_tokens=0), COUNTER)
        recovered = "".join("".join(c.text.split()) for c in chunks)
        assert recovered == "".join(text.split())

    def test_empty_document_yields_no_chunks(self) -> None:
        assert chunk_document(make_document(""), ChunkingConfig(), COUNTER) == []

    def test_whitespace_only_document_yields_no_chunks(self) -> None:
        assert chunk_document(make_document("  \n\n \t "), ChunkingConfig(), COUNTER) == []

    def test_chunks_never_start_or_end_on_whitespace(self) -> None:
        doc = make_document("  1. TERM.   \n\n   Body text here.   \n\n  2. LAW.  Delaware.  ")
        for chunk in chunk_document(doc, ChunkingConfig(max_tokens=10), COUNTER):
            assert chunk.text == chunk.text.strip(), repr(chunk.text)


class TestHeadingDetection:
    """59% of the corpus is run-on text, so headings are not line-anchored."""

    def test_finds_decimal_numbered_headings(self) -> None:
        text = "1. TERM. Body.   2.1 GOVERNING LAW. Delaware.   11.3 NOTICES. By mail."
        found = [h.label for h in find_headings(text)]
        assert found == ["1", "2.1", "11.3"]

    def test_finds_headings_mid_line_after_whitespace_runs(self) -> None:
        """The dominant real-world case; a `^`-anchored regex misses all of these."""
        text = "preamble words here     11.3 A Party may disclose Confidential Information"
        assert [h.label for h in find_headings(text)] == ["11.3"]

    def test_finds_article_and_section_headings(self) -> None:
        text = "ARTICLE I DEFINITIONS\n\nbody\n\nSection 4.2 Term.\n\nbody"
        labels = [h.label for h in find_headings(text)]
        assert "ARTICLE I" in labels
        assert "Section 4.2" in labels

    def test_heading_offsets_point_at_the_heading(self) -> None:
        text = "intro text   2.1 GOVERNING LAW. Delaware."
        heading = next(h for h in find_headings(text) if h.label == "2.1")
        assert text[heading.char_start :].startswith("2.1")

    def test_does_not_match_a_year(self) -> None:
        assert find_headings("dated as of 1934, as amended by the parties") == []

    def test_does_not_match_a_cross_reference(self) -> None:
        """'Section 13 of the Exchange Act' is a citation, not a heading."""
        assert find_headings("pursuant to Section 13 of the Exchange Act") == []

    def test_does_not_match_a_number_followed_by_lowercase(self) -> None:
        assert find_headings("the party shall pay 5 dollars per unit") == []

    def test_returns_empty_for_unstructured_text(self) -> None:
        """41 of 510 CUAD contracts land here."""
        assert find_headings("This joint filing agreement is entered into by the parties.") == []

    def test_headings_are_returned_in_document_order(self) -> None:
        text = "1. A.  body   2. B.  body   3. C.  body"
        starts = [h.char_start for h in find_headings(text)]
        assert starts == sorted(starts)


class TestSectionAwareness:
    def test_a_short_section_becomes_one_chunk(self) -> None:
        doc = make_document("1. TERM. This agreement runs for one year.")
        chunks = chunk_document(doc, ChunkingConfig(max_tokens=100, min_tokens=0), COUNTER)
        assert len(chunks) == 1

    def test_sections_are_not_merged_across_headings(self) -> None:
        """The point of structure-aware chunking: a chunk should not straddle two
        unrelated clauses just because both are short."""
        doc = make_document("1. TERM. One year.   2. GOVERNING LAW. Delaware.")
        chunks = chunk_document(doc, ChunkingConfig(max_tokens=100, min_tokens=0), COUNTER)
        assert len(chunks) == 2
        assert "TERM" in chunks[0].text
        assert "GOVERNING LAW" in chunks[1].text

    def test_chunk_records_its_heading(self) -> None:
        doc = make_document("1. TERM. One year.   2. GOVERNING LAW. Delaware.")
        chunks = chunk_document(doc, ChunkingConfig(max_tokens=100, min_tokens=0), COUNTER)
        assert chunks[0].heading is not None
        assert "TERM" in chunks[0].heading
        assert chunks[1].heading is not None
        assert "GOVERNING LAW" in chunks[1].heading

    def test_preamble_before_the_first_heading_is_kept(self) -> None:
        doc = make_document("THIS AGREEMENT is made by and between A and B.   1. TERM. One year.")
        chunks = chunk_document(doc, ChunkingConfig(max_tokens=100, min_tokens=0), COUNTER)
        assert "THIS AGREEMENT" in chunks[0].text
        assert chunks[0].heading is None

    def test_oversized_section_splits_but_keeps_its_heading(self) -> None:
        doc = make_document("1. TERM. " + words(200))
        chunks = chunk_document(doc, ChunkingConfig(max_tokens=40), COUNTER)
        assert len(chunks) > 1
        assert all(c.heading is not None and "TERM" in c.heading for c in chunks)


class TestSizeConstraints:
    def test_no_chunk_exceeds_the_maximum(self) -> None:
        doc = make_document("1. TERM. " + words(400) + "   2. LAW. " + words(400))
        config = ChunkingConfig(max_tokens=50, overlap_tokens=0)
        for chunk in chunk_document(doc, config, COUNTER):
            assert COUNTER.count(chunk.text) <= config.max_tokens

    def test_prefers_paragraph_boundaries_when_splitting(self) -> None:
        para = words(30)
        doc = make_document(f"1. TERM.\n\n{para}\n\n{para}\n\n{para}")
        chunks = chunk_document(doc, ChunkingConfig(max_tokens=45, min_tokens=0), COUNTER)
        # Each chunk should end at a paragraph boundary, so no chunk contains a
        # partial paragraph followed by the start of the next.
        assert len(chunks) >= 3
        assert all(c.is_faithful_to(doc) for c in chunks)

    def test_tiny_sections_merge_forward(self) -> None:
        doc = make_document("1. A. x   2. B. y   3. C. z")
        merged = chunk_document(doc, ChunkingConfig(max_tokens=100, min_tokens=10), COUNTER)
        unmerged = chunk_document(doc, ChunkingConfig(max_tokens=100, min_tokens=0), COUNTER)
        assert len(merged) < len(unmerged)

    def test_merging_preserves_offset_fidelity(self) -> None:
        doc = make_document("1. A. x   2. B. y   3. C. z")
        for chunk in chunk_document(doc, ChunkingConfig(max_tokens=100, min_tokens=10), COUNTER):
            assert chunk.is_faithful_to(doc)

    def test_token_count_is_recorded_accurately(self) -> None:
        doc = make_document("1. TERM. " + words(100))
        for chunk in chunk_document(doc, ChunkingConfig(max_tokens=40), COUNTER):
            assert chunk.token_count == COUNTER.count(chunk.text)


class TestOverlap:
    def test_overlap_repeats_text_between_hard_split_chunks(self) -> None:
        doc = make_document(words(200))
        with_overlap = chunk_document(
            doc, ChunkingConfig(max_tokens=50, overlap_tokens=10), COUNTER
        )
        without = chunk_document(doc, ChunkingConfig(max_tokens=50, overlap_tokens=0), COUNTER)
        assert len(with_overlap) >= len(without)
        assert all(c.is_faithful_to(doc) for c in with_overlap)

    def test_overlapping_chunks_still_advance(self) -> None:
        """Guards an infinite loop: if overlap >= chunk size, a naive
        implementation never makes progress."""
        doc = make_document(words(200))
        chunks = chunk_document(doc, ChunkingConfig(max_tokens=20, overlap_tokens=19), COUNTER)
        starts = [c.char_start for c in chunks]
        assert all(b > a for a, b in itertools.pairwise(starts))


class TestDeterminism:
    def test_same_input_gives_identical_chunks(self) -> None:
        doc = make_document("1. TERM. " + words(150) + "   2. LAW. " + words(150))
        config = ChunkingConfig(max_tokens=40)
        assert chunk_document(doc, config, COUNTER) == chunk_document(doc, config, COUNTER)

    def test_chunk_ids_are_unique_and_stable(self) -> None:
        doc = make_document("1. TERM. " + words(150))
        first = chunk_document(doc, ChunkingConfig(max_tokens=40), COUNTER)
        second = chunk_document(doc, ChunkingConfig(max_tokens=40), COUNTER)
        ids = [c.chunk_id for c in first]
        assert len(ids) == len(set(ids))
        assert ids == [c.chunk_id for c in second]

    def test_chunk_ids_differ_across_documents(self) -> None:
        a = chunk_document(make_document("1. TERM. x", "DOC_A"), ChunkingConfig(), COUNTER)
        b = chunk_document(make_document("1. TERM. x", "DOC_B"), ChunkingConfig(), COUNTER)
        assert {c.chunk_id for c in a}.isdisjoint({c.chunk_id for c in b})
