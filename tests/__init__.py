"""Tests. The agent logger is disabled so unit tests never write into logs/agent.log."""
import logging

logging.getLogger("finance_agent").disabled = True
