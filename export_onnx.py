import os
import torch
from stable_baselines3 import PPO


SB3_MODEL_PATH = "models/luobo_cat_ppo.zip"
ONNX_EXPORT_PATH = "models/luobo_cat_policy.onnx"


class ActorOnly(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self.policy.get_distribution(obs)
        action = dist.get_actions(deterministic=True)
        return action.to(torch.int64)


def main():
    if not os.path.exists(SB3_MODEL_PATH):
        raise FileNotFoundError(
            f"找不到模型文件: {SB3_MODEL_PATH}\n"
            f"请先训练并保存: model.save('models/luobo_cat_ppo')"
        )

    os.makedirs(os.path.dirname(ONNX_EXPORT_PATH), exist_ok=True)

    model = PPO.load(SB3_MODEL_PATH, device="cpu")

    policy = model.policy
    policy.eval()
    policy.to("cpu")

    actor = ActorOnly(policy)
    actor.eval()
    actor.to("cpu")

    dummy_obs = torch.tensor([[0.0]], dtype=torch.float32, device="cpu")

    torch.onnx.export(
        actor,
        dummy_obs,
        ONNX_EXPORT_PATH,
        input_names=["obs"],
        output_names=["action"],
        opset_version=18,
        dynamic_axes={
            "obs": {0: "batch"},
            "action": {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    print("✅ 导出完成！")
    print(f"- ONNX 文件: {ONNX_EXPORT_PATH}")
    print("- input:  obs  (float32, shape=[batch,1])")
    print("- output: action (int64, shape=[batch])")
    print("\n提示：如果你要在浏览器跑，请用 onnxruntime-web 加载这个 .onnx 文件。")


if __name__ == "__main__":
    main()
