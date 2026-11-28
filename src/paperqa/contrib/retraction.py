from __future__ import annotations

import logging
import re
from collections.abc import Callable, Collection, Sequence
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from paperqa.types import Context, Doc, DocDetails, Text

logger = logging.getLogger(__name__)

RETRACTION_WARNING_TAG = "[RETRACTED PAPER WARNING]"

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

URL_PREFIXES_TO_REMOVE = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)

# Landmark retracted DOIs catalog (canonical lowercased)
LANDMARK_RETRACTED_DOIS: set[str] = {
    # Wakefield et al. (Lancet 1998) - Autism / MMR vaccine fraud
    "10.1016/s0140-6736(97)11096-0",
    # Mehra et al. (Lancet 2020) - Surgisphere hydroxychloroquine COVID-19
    "10.1016/s0140-6736(20)31180-6",
    # Mehra et al. (NEJM 2020) - Surgisphere cardiovascular disease COVID-19
    "10.1056/nejmoa2012410",
    # Obokata et al. (Nature 2014) - STAP stem cells
    "10.1038/nature12968",
    "10.1038/nature12969",
    # Hwang Woo-suk et al. (Science 2004, 2005) - Human embryonic stem cells
    "10.1126/science.1094515",
    "10.1126/science.1112286",
    # Jan Hendrik Schön et al. (Science 2000 / Nature 2001) - Molecular crystals / superconductivity
    "10.1126/science.288.5464.233",
    "10.1038/35069008",
    # Paolo Macchiarini et al. (Lancet 2011) - Synthetic trachea transplant
    "10.1016/s0140-6736(11)60071-2",
    # Séralini et al. (Food and Chemical Toxicology 2012) - Roundup & GM maize toxicity
    "10.1016/j.fct.2012.08.005",
    # Scott Reuben et al. (Anesthesiology / Anesth Analg) - Fabricated clinical trials
    "10.1097/00000542-200507000-00028",
    # Dong-Pyou Han (Nature / AIDS) - Fabricated rabbit HIV vaccine antibodies
    "10.1038/nature07286",
    # Anil Potti et al. (NEJM 2006 / Nat Med 2006) - Microarray genomics for cancer therapy
    "10.1056/nejmoa052167",
    "10.1038/nm1491",
    # Silvia Bulfone-Paus et al. (JEM 2007) - Immune cell biology manipulations
    "10.1084/jem.20070659",
    # Olivier Voinnet et al. (Science 2004) - Plant RNA silencing
    "10.1126/science.1095835",
}


def clean_doi(raw_doi: str) -> str:
    """Normalize a DOI string by removing URL prefixes, whitespace, and lowercasing."""
    cleaned = raw_doi.strip()
    for prefix in URL_PREFIXES_TO_REMOVE:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned.lower()


def is_retracted_doi(
    doi: str,
    custom_dois: Collection[str] | None = None,
    hook: Callable[[str], bool] | None = None,
    include_landmarks: bool = True,
) -> bool:
    """Check if a given DOI corresponds to a retracted paper.

    Args:
        doi: The DOI string to check (can be raw DOI or DOI URL).
        custom_dois: Optional set/collection of additional retracted DOIs.
        hook: Optional callable hook taking the cleaned DOI and returning bool.
        include_landmarks: Whether to check the built-in landmark retracted DOIs catalog.

    Returns:
        True if the DOI is known to be retracted, False otherwise.
    """
    cleaned = clean_doi(doi)
    if hook is not None and hook(cleaned):
        return True
    if custom_dois:
        normalized_custom = {clean_doi(d) for d in custom_dois}
        if cleaned in normalized_custom:
            return True
    return bool(include_landmarks and cleaned in LANDMARK_RETRACTED_DOIS)


def extract_doi_from_target(item: Any) -> str | None:
    """Extract a cleaned DOI from a string, Doc, DocDetails, or Text."""
    if isinstance(item, str):
        match = DOI_PATTERN.search(item)
        if match:
            return clean_doi(match.group(0))
        return clean_doi(item) if "10." in item else None

    # Check doc object or text.doc
    doc = getattr(item, "doc", item)
    for field in ("doi", "citation", "docname"):
        val = getattr(doc, field, None)
        if isinstance(val, str):
            match = DOI_PATTERN.search(val)
            if match:
                return clean_doi(match.group(0))
            if field == "doi" and val.strip():
                return clean_doi(val)
    return None


class RetractionFilter:
    """Filter to detect, exclude, or flag retracted papers and chunks."""

    def __init__(
        self,
        action: Literal["drop", "warn"] = "drop",
        custom_dois: Collection[str] | None = None,
        is_retracted_hook: Callable[[str], bool] | None = None,
        include_landmarks: bool = True,
    ) -> None:
        self.action: Literal["drop", "warn"] = action
        self.custom_dois: set[str] = {clean_doi(d) for d in (custom_dois or ())}
        self.is_retracted_hook = is_retracted_hook
        self.include_landmarks = include_landmarks

    @classmethod
    def from_settings(cls, settings: Any) -> RetractionFilter | None:
        """Create a RetractionFilter instance from settings if enabled, or None."""
        evidence_retraction_filter = getattr(
            settings, "evidence_retraction_filter", False
        )
        answer_config = getattr(settings, "answer", None)
        if answer_config is not None:
            evidence_retraction_filter = (
                evidence_retraction_filter
                or getattr(answer_config, "evidence_retraction_filter", False)
            )

        if not evidence_retraction_filter:
            return None

        retraction_action: Literal["drop", "warn"] = "drop"
        if getattr(settings, "retraction_action", None) == "warn" or (
            answer_config is not None
            and getattr(answer_config, "retraction_action", None) == "warn"
        ):
            retraction_action = "warn"
        elif getattr(settings, "retraction_action", None) in {"drop", "warn"}:
            retraction_action = settings.retraction_action
        elif answer_config is not None and getattr(
            answer_config, "retraction_action", None
        ) in {"drop", "warn"}:
            retraction_action = answer_config.retraction_action

        return cls(action=retraction_action)

    def apply_to_matches(self, matches: Sequence[Text]) -> list[Text]:
        """Apply the configured retraction action to evidence matches."""
        if self.action == "drop":
            return [m for m in matches if not self.is_retracted(m)]
        return self.filter_texts(matches, action="warn")

    def apply_to_contexts(self, contexts: Sequence[Context]) -> list[Context]:
        """Apply the configured retraction action to gathered contexts."""
        return self.filter_contexts(contexts)

    def is_retracted(self, item: str | Doc | DocDetails | Text | Any) -> bool:
        """Check if an item (DOI string, Doc, DocDetails, Text) is retracted."""
        # Check explicit flag if set on Doc / DocDetails / Text / Context
        if getattr(item, "is_retracted", False) is True:
            return True

        doc = getattr(item, "doc", None)
        if doc is not None and getattr(doc, "is_retracted", False) is True:
            return True

        doi = extract_doi_from_target(item)
        if doi:
            return is_retracted_doi(
                doi,
                custom_dois=self.custom_dois,
                hook=self.is_retracted_hook,
                include_landmarks=self.include_landmarks,
            )
        return False

    def filter_texts(
        self,
        texts: Sequence[Text],
        action: Literal["drop", "warn"] | None = None,
    ) -> list[Text]:
        """Apply retraction policy to a list of Text chunks.

        If action is 'drop', retracted chunks are removed.
        If action is 'warn', retracted chunks are tagged with a warning and metadata.
        """
        chosen_action = action or self.action
        output: list[Text] = []
        for t in texts:
            if self.is_retracted(t):
                if chosen_action == "drop":
                    continue
                if chosen_action == "warn":
                    warned_t = t.model_copy(deep=True)
                    if not warned_t.text.startswith(RETRACTION_WARNING_TAG):
                        warned_t.text = f"{RETRACTION_WARNING_TAG} {warned_t.text}"
                    warned_t.is_retracted = True
                    warned_t.retraction_warning = RETRACTION_WARNING_TAG
                    output.append(warned_t)
                    continue
            output.append(t)
        return output

    def filter_contexts(
        self,
        contexts: Sequence[Context],
        action: Literal["drop", "warn"] | None = None,
    ) -> list[Context]:
        """Apply retraction policy to a list of Contexts."""
        chosen_action = action or self.action
        output: list[Context] = []
        for c in contexts:
            if self.is_retracted(c) or self.is_retracted(c.text):
                if chosen_action == "drop":
                    continue
                if chosen_action == "warn":
                    warned_c = c.model_copy(deep=True)
                    if not warned_c.context.startswith(RETRACTION_WARNING_TAG):
                        warned_c.context = f"{RETRACTION_WARNING_TAG} {warned_c.context}"
                    if not warned_c.text.text.startswith(RETRACTION_WARNING_TAG):
                        warned_c.text.text = f"{RETRACTION_WARNING_TAG} {warned_c.text.text}"
                    warned_c.is_retracted = True
                    warned_c.retraction_warning = RETRACTION_WARNING_TAG
                    warned_c.text.is_retracted = True
                    warned_c.text.retraction_warning = RETRACTION_WARNING_TAG
                    output.append(warned_c)
                    continue
            output.append(c)
        return output
