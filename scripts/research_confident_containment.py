"""Compose existing confidence NMS and confidence-ranked nested suppression."""
from scripts.compare_balloon_containment import confidence_ordered
from scripts.research_balloon_confidence_selection import confidence_selection


def select(found):
    """No new predictions: only reverse final tiny-part suppression when stronger."""
    return confidence_ordered(confidence_selection(found))
