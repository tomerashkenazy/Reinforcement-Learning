import torch
envs = ["Acrobot-v1", "CartPole-v1", "MountainCar-v0"]
for env in envs:
    actor_path = f"Assignment_3/transfer_models/actor_critic_{env}/actor_net.pth"
    critic_path = f"Assignment_3/transfer_models/actor_critic_{env}/critic_net.pth"
    actor_ckpt = torch.load(actor_path, map_location="cpu")
    critic_ckpt = torch.load(critic_path, map_location="cpu")
    print(f"\n=== {env} Actor Network ===")
    for k, v in actor_ckpt.items():
        print(f"{k:20s} {tuple(v.shape)}")
    print(f"\n=== {env} Critic Network ===")
    for k, v in critic_ckpt.items():
        print(f"{k:20s} {tuple(v.shape)}")

