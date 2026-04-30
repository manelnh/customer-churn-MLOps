"""Compatibility entry point for the project's unit test suite."""

import unittest


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover('tests')
    runner = unittest.TextTestRunner(verbosity=2)
    raise SystemExit(0 if runner.run(suite).wasSuccessful() else 1)
