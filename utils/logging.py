import mlflow


class Logger(object):
    """ Class for logging experiment to mlflow """
    def __init__(self, cfg):
        self.cfg = cfg
        self.start_logging()

    def start_logging(self, experiment_name=None):
        """
        Starts mlflow logging
        :param experiment_name: experiment name for mlflow visualization
        """
        if self.cfg.log_metrics:
            experiment_name = self.cfg.experiment_name if experiment_name is None else experiment_name
            mlflow.start_run(run_name=experiment_name)

    def end_logging(self):
        """
        Finishes mlflow logging
        """
        if self.cfg.log_metrics:
            mlflow.end_run()

    def log_metrics(self, names, metrics, step):
        """
        Logs metrics in given list with corresponding names
        :param names: list of names of given metrics
        :param metrics: list of given metrics
        :param step: step to log
        """
        if self.cfg.log_metrics:
            for name, metric in zip(names, metrics):
                mlflow.log_metric(name, metric, step)

    def log_params(self):
        """
        Logs experiment config with all parameters
        """
        if self.cfg.log_metrics:
            mlflow.log_param('cfg', self.cfg)
