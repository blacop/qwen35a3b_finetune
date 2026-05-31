from __future__ import annotations


class CalculatorError(Exception):
    code = "INTERNAL_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidInputError(CalculatorError):
    code = "INVALID_INPUT"


class MissingFieldError(CalculatorError):
    code = "MISSING_FIELD"


class UnsupportedFormulaError(CalculatorError):
    code = "UNSUPPORTED_FORMULA"

