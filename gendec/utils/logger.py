class TrainingConsoleLogger:
    def __init__(self, disable_tqdm=False):
        self.disable_tqdm = bool(disable_tqdm)

    def format_metrics(self, metrics):
        if not metrics:
            return "none"
        items = []
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, (int, float)):
                items.append(f"{key}={float(value):.4f}")
            else:
                items.append(f"{key}={value}")
        return ", ".join(items)

    def progress_bar(self, iterable, *, desc, leave=False):
        try:
            from tqdm.auto import tqdm
        except ModuleNotFoundError:
            return iterable
        return tqdm(iterable, desc=desc, leave=leave, disable=self.disable_tqdm)

    def update_progress_postfix(self, progress, metrics):
        if hasattr(progress, "set_postfix") and metrics:
            progress.set_postfix(
                {
                    key: f"{float(value):.4f}" if isinstance(value, (int, float)) else value
                    for key, value in metrics.items()
                }
            )

    def print_epoch_summary(self, epoch, num_epochs, train_metrics, val_metrics, sample_metrics):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  train: {self.format_metrics(train_metrics)}")
        print(f"  val: {self.format_metrics(val_metrics)}")
        print(f"  samples: {self.format_metrics(sample_metrics)}")
