class CometMLLogger:
    def __init__(self,
                 project_name: str,
                 exp_name: str,
                 exp_config: dict,
                 exp_dir: str,
                 checkpoint_dir: str,
                 is_resuming: bool,
                 epoch_length: int,
                 workspace: str = None):
        self.epoch_length = epoch_length
        self.exp_name = exp_name

        if is_resuming:
            with open(os.path.join(checkpoint_dir, "comet.exp_key"), "r") as f:
                prev_exp_key = f.readline().strip()
            self.experiment = comet_ml.ExistingExperiment(previous_experiment=prev_exp_key,
                                                          auto_metric_logging=False,
                                                          log_env_details=True,
                                                          log_env_host=True,
                                                          log_code=False)
        else:
            self.experiment = comet_ml.Experiment(project_name=project_name,
                                                  workspace=workspace,
                                                  auto_metric_logging=False,
                                                  log_env_details=True,
                                                  log_env_host=True,
                                                  log_code=False)
            self.experiment.set_name(self.exp_name)
            for f in glob(f'{exp_dir}/*.py'):
                self.experiment.log_code(file_name=f)
            self._log_config(exp_config)

            with open(os.path.join(checkpoint_dir, "comet.exp_key"), "w") as f:
                f.write(self.experiment.get_key())

            with open(os.path.join(checkpoint_dir, "splits.pickle.sha256"), "r") as f:
                splits_hash = f.readline().strip()
                self.experiment.log_parameter("splits_hash", splits_hash)

    def _log_config(self, config):
        def recurse(config_prefix, config):
            for key, value in config.items():
                key = "->".join(filter(None, [config_prefix, key]))
                if isinstance(value, dict):
                    recurse(key, value)
                else:
                    self.experiment.log_parameter(key, value)
        recurse("", config)

    def attach(self, engine):
        if not isinstance(engine, Engine):
            raise TypeError(f"Argument engine should be ignite.engine.Engine, but given {type(engine)}")

        if not engine.has_event_handler(self.on_epoch):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.on_epoch)

        if not engine.has_event_handler(self.on_terminate):
            engine.add_event_handler(Events.TERMINATE, self.on_terminate)

    def on_epoch(self, engine):
        self.experiment.log_metrics(engine.state.metrics, step=engine.state.epoch * self.epoch_length, epoch=engine.state.epoch)

    def on_terminate(self, engine):
        self.experiment.add_tag("TERMINATED")
        self.experiment.send_notification(title=f"TERMINATED EXPERIMENT - {self.exp_name}",
                                          status="aborted")
