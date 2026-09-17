import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('signed_judge_api',Path(__file__).with_name('judge-api.py'))
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)

def test_judge_model_requests_use_local_gateway():
    import asyncio
    registry=api.GatewayOnlyRegistry('quokka', 'http://127.0.0.1:8000')
    assert asyncio.run(registry.sample_servers('quokka')) == [('http://127.0.0.1:8000',1.0)]
    with pytest.raises(ValueError):
        asyncio.run(registry.sample_servers('other'))

def test_quokka_uses_terminal_training_template():
    profile=json.loads(Path(__file__).with_name('profiles').joinpath('quokka9b.json').read_text())
    assert profile['rubric_mode']=='per_rubric'
    assert profile['tools']==['webterminal']
    template=Path(profile['prompt_template_path']).read_text()
    assert 'terminal' in template and 'standard input' in template

def test_empty_and_truncated_outputs_cannot_escape_signed_penalties():
    import asyncio
    from primebeaker.environments.jtc_rubrichub_judge_env import JTCJudgeRubric
    class NoJudge:
        async def verify_output(self,**kwargs):raise AssertionError('Should not call judge')
    rubric=JTCJudgeRubric(judge_client=NoJudge(),judge_model_path='quokka-test',non_termination_penalty=-1.25,invalid_output_penalty=-1.25)
    answer=json.dumps({'record_id':'test','rubrics':[{'text':'Bad claim','weight':-3}]})
    for truncated in [False,True]:
        state={'answer':answer,'prompt':'Question','completion':[],'is_truncated':truncated}
        asyncio.run(rubric.score_rollout(state))
        assert state['reward']==-1.25
