"""Custom exceptions for Village AI War."""


class VillageWarError(Exception):
    """Base exception for domain errors in Village AI War."""

    pass


class InvalidActionError(VillageWarError):
    """Raised when an action is structurally or logically invalid."""

    pass


class InsufficientResourcesError(VillageWarError):
    """Raised when an operation requires more resources than available."""

    pass
