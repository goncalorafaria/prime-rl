import os
from pathlib import Path
import pytest
os.chdir(Path(__file__).resolve().parents[2])
os.environ['LITECAST_TEST_REDIS_SERVER']='/gscratch/ark/graf/redis-stable/src/redis-server'
raise SystemExit(pytest.main(['-q','--confcutdir=tests/unit/litecast',
 'tests/unit/litecast/test_two_middles.py','tests/unit/litecast/test_protocol.py']))
