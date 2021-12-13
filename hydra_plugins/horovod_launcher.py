import logging
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import cloudpickle
from hydra.core.config_store import ConfigStore
from hydra.core.singleton import Singleton
from hydra.core.utils import filter_overrides
from hydra.core.utils import JobReturn
from hydra.core.utils import run_job
from hydra.core.utils import setup_globals
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext
from hydra.types import TaskFunction
from omegaconf import DictConfig
from omegaconf import open_dict


log = logging.getLogger(__name__)


@dataclass
class LauncherConfig:
    _target_: str = 'hydra_plugins.horovod_launcher.HorovodLauncher'


ConfigStore.instance().store(
    group='hydra/launcher',
    name='horovod',
    node=LauncherConfig,
)


class HorovodLauncher(Launcher):

    def __init__(self) -> None:
        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.sweep_configs: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

    def setup(self, *, hydra_context: HydraContext, task_function: TaskFunction, config: DictConfig) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function
        # make sure we can pickle the hydra function!
        try:
            cloudpickle.dumps(self.task_function)
        except Exception as e:
            raise RuntimeError(f'Cannot pickle the hydra function: {self.task_function}') from e


    def _run(self, sweep_overrides: List[str], job_dir_key: str, job_num: int, job_id: str, singleton_state: Dict[type, Singleton]) -> JobReturn:
        assert self.hydra_context is not None
        assert self.config is not None
        assert self.task_function is not None

        Singleton.set_state(singleton_state)
        setup_globals()
        sweep_config = self.hydra_context.config_loader.load_sweep_config(self.config, sweep_overrides)

        with open_dict(sweep_config.hydra.job) as job:
            # Populate new job variables
            job.id = job_id
            sweep_config.hydra.job.num = job_num

        return run_job(
            hydra_context=self.hydra_context,
            task_function=self.task_function,
            config=sweep_config,
            job_dir_key=job_dir_key,
            job_subdir_key="hydra.sweep.subdir",
        )

    def launch(self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int) -> Sequence[JobReturn]:
        assert self.hydra_context is not None
        assert self.config is not None
        assert self.task_function is not None

        num_jobs = len(job_overrides)
        assert num_jobs > 0

        # initialise the sweet directory!
        log.info(f"Horovod sweep output dir : {self.config.hydra.sweep.dir}")
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        # launch all the jobs
        results = []
        for idx, overrides in list(enumerate(job_overrides)):
            # get the job ID
            idx = initial_job_idx + idx

            # print the overrides
            lst = " ".join(filter_overrides(overrides))
            log.info(f"\t#{idx} : {lst}")

            # make the task
            task_fn = partial(
                self._run,
                sweep_overrides=list(overrides),
                job_dir_key='hydra.sweep.dir',
                job_num=int(idx),
                job_id=f'job_id_for_{idx}',
                singleton_state=Singleton.get_state(),
            )

            # run the task
            result = self._launch_horovod_task(sweep_dir=str(sweep_dir), job_num=idx, task_fn=task_fn)
            results.append(result)

        # done!
        return results

    @staticmethod
    def _launch_horovod_task(sweep_dir: str, job_num: int, task_fn: callable) -> JobReturn:
        import subprocess
        import torch
        import sys
        import os
        import cloudpickle
        from mtg_ml import __file__ as mtg_path

        # get the runner path
        runner_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(mtg_path)), 'hydra_plugins/_horovod_runner.py'))

        # pickle the function
        pickled_fn_path = os.path.join(sweep_dir, f'task_{job_num}.pkl')
        pickled_result_path = os.path.join(sweep_dir, f'task-result_{job_num}.pkl')
        with open(pickled_fn_path, 'wb') as pickle_file:
            cloudpickle.dump((task_fn, pickled_result_path), pickle_file)

        # generate command
        n_gpus = torch.cuda.device_count()
        command = [
            'horovodrun', '-np', f'{n_gpus}', '-H', f'localhost:{n_gpus}',
            sys.executable, runner_path, pickled_fn_path,
        ]

        # debug
        log.info(f'Running Task With Horovod on {n_gpus} GPUs:')
        log.info(command)

        # run command
        process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()

        try:
            with open(pickled_result_path, 'rb') as pickle_file:
                job_return = cloudpickle.load(pickle_file)
            assert isinstance(job_return, JobReturn)
            return job_return
        except:
            log.warning(f'Job failed to return result at: {repr(pickled_result_path)}')
            return None
