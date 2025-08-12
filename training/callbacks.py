from stable_baselines3.common.callbacks import BaseCallback

class AutoSaveCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.model.save(self.save_path)
            if self.verbose:
                print(f"💾 Автосохранение модели на шаге {self.num_timesteps} → {self.save_path}")
        return True
