import argparse
import sys
from configs.capture_config import CaptureConfig
from training.train import train_model
from training.resume import resume_training
from training.play import play_model

# Импортируем оба варианта окружений
from envs.gd_env_soft import GeometryDashLiveEnv as SoftEnv
from envs.gd_env_strict import GeometryDashLiveEnv as StrictEnv


def main():
    parser = argparse.ArgumentParser(description="Geometry Dash RL Trainer/Player")

    parser.add_argument("--mode", choices=["train", "resume", "play"], required=True,
                        help="Режим запуска: train | resume | play")
    parser.add_argument("--env", choices=["soft", "strict"], default="soft",
                        help="Выбор окружения: soft (с фиксами) или strict (оригинал)")

    parser.add_argument("--left", type=int, required=True, help="screen capture left")
    parser.add_argument("--top", type=int, required=True, help="screen capture top")
    parser.add_argument("--width", type=int, required=True, help="screen capture width")
    parser.add_argument("--height", type=int, required=True, help="screen capture height")
    parser.add_argument("--fps", type=int, default=30, help="target capture fps")
    parser.add_argument("--model", type=str, default="models/gd_live_ppo.zip",
                        help="Путь для сохранения/загрузки модели")

    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="Количество шагов для обучения или дообучения")

    # Параметры для resume (при train игнорируются)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=8)
    parser.add_argument("--ent_coef", type=float, default=0.008)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=0.03)
    parser.add_argument("--save_freq", type=int, default=2500)

    args = parser.parse_args()

    # Определяем окружение
    env_class = SoftEnv if args.env == "soft" else StrictEnv

    # Конфиг захвата экрана
    capture_cfg = CaptureConfig(
        left=args.left,
        top=args.top,
        width=args.width,
        height=args.height,
        target_fps=args.fps
    )

    # Режимы работы
    if args.mode == "train":
        print(f"🚀 Запуск обучения ({args.env}) на {args.timesteps} шагов")
        train_model(env_class, capture_cfg,
                    total_timesteps=args.timesteps,
                    show_window=False,
                    model_path=args.model)

    elif args.mode == "resume":
        print(f"🔄 Возобновление обучения ({args.env}) на {args.timesteps} шагов")
        resume_training(env_class,
                        capture_cfg=capture_cfg,
                        model_path=args.model,
                        total_timesteps=args.timesteps,
                        save_freq=args.save_freq,
                        lr=args.lr,
                        n_steps=args.n_steps,
                        batch_size=args.batch_size,
                        n_epochs=args.n_epochs,
                        ent_coef=args.ent_coef,
                        gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                        clip_range=args.clip_range,
                        vf_coef=args.vf_coef,
                        max_grad_norm=args.max_grad_norm,
                        target_kl=args.target_kl)

    elif args.mode == "play":
        print(f"🎮 Запуск игры в режиме {args.env}")
        play_model(env_class, capture_cfg, model_path=args.model)

    else:
        print("❌ Неизвестный режим запуска.")
        sys.exit(1)


if __name__ == "__main__":
    main()
