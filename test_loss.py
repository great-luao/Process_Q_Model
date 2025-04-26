import torch
import torch.nn as nn

LOSS_TYPE = 'rank'  # 'nce', 'mse', 'orm', 'rank'
zeta = 4

def ranking_loss(rewards,labels,has_neg):
    """
    rewards: 模型在每个token位置预测的值, B*S
    labels: B*S
    has_neg: B, 有neg标签就为1, 没有就是0.
    """
    pos_rewards_exp = torch.where(labels == 1, (rewards).exp(), 0) # Q_c
    neg_rewards_exp = torch.where(labels == 0, (rewards+zeta).exp(), 0).flip(dims=[-1]) # Q_w,越靠后的错误reward应该越小
    neg_reward_sum = neg_rewards_exp.sum(-1)

    pos_rewards_cumsum = torch.cat([torch.zeros(rewards.shape[0], 1, device=rewards.device).exp(), pos_rewards_exp],
                                    dim=1).cumsum(-1)[:, :-1]
    pos_rewards_cumsum = torch.cat([torch.zeros(rewards.shape[0], 1, device=rewards.device), pos_rewards_cumsum],
                                    dim=-1)

    reward_exp_cur = torch.where(labels == 1, pos_rewards_exp, 1)
    reward_exp_cur = torch.cat([torch.zeros(rewards.shape[0], 1, device=rewards.device).exp(), reward_exp_cur], dim=-1)

    # ?这里是不是多+了一个reward_exp_cur?
    loss = -torch.log(reward_exp_cur / (reward_exp_cur + pos_rewards_cumsum + neg_reward_sum[..., None] + 1e-5))

    labels = torch.cat([has_neg[..., None], labels], dim=-1)
    loss = (torch.where(labels == 1, loss, 0).sum(-1) / torch.where(labels == 1, 1, 0).sum(-1)).mean()
    return loss

# Example usage
if __name__ == "__main__":
    # B, S = 2, 5
    # labels = torch.tensor([[1, 2, 3, -100, -100], [1, 2, -100, -100, -100]], dtype=torch.float32)
    # # 创建非 -100 的掩码
    # result = []
    # for row in labels:
    #     valid = row[row != -100]
    #     result.append(valid[-1].unsqueeze(0))
    # result = torch.stack(result)

    # print(result)  # tensor([[3], [2]])

    loss_true = torch.tensor([2.0, 2.0, 2.0, 2.0])
    loss_false = torch.tensor([-2.0, -2.0, -2.0, -2.0])
    correctness = torch.tensor([1., 0., 1.0, 0.])

    loss_all = torch.where(correctness.bool(), loss_true, loss_false)
    print(loss_all)


# def compute_loss(inputs):

#     rewards = 

#     if LOSS_TYPE == 'nce' or LOSS_TYPE == 'orm':
#         loss_fn = nn.BCELoss(reduction='none')
#     elif LOSS_TYPE=='mse':
#         loss_fn = nn.MSELoss(reduction='none')

#     if LOSS_TYPE=='nce':
#         rewards = rewards.gather(dim=-1, index=inputs['special_tokens'])
#         rewards = rewards.sigmoid()
#         loss = (loss_fn(reduction='none')(rewards, torch.where(inputs['step_labels']!=-100,inputs['step_labels'],0).bfloat16()) * (inputs['step_labels']!=-100)).sum()/(inputs['step_labels']!=-100).sum()
#     elif LOSS_TYPE=='mse':
#         rewards = rewards.gather(dim=-1, index=inputs['special_tokens'])
#         rewards = rewards.sigmoid()
#         loss = (loss_fn(rewards,
#                                 torch.where(inputs['step_labels'] != -100, inputs['step_labels'], 0).bfloat16()) * (
#                             inputs['step_labels'] != -100)).sum() / (inputs['step_labels'] != -100).sum()
#     elif LOSS_TYPE=='orm':
#         rewards = rewards.gather(dim=-1, index=inputs['orm_tokens'][...,None])
#         rewards = rewards.sigmoid()
#         loss = loss_fn(rewards.squeeze(1),1-inputs['has_neg'].bfloat16()).mean()
#     elif LOSS_TYPE=='rank':
#         rewards = rewards.gather(dim=-1, index=inputs['special_tokens'])
#         loss = ranking_loss(rewards,inputs['step_labels'],inputs['has_neg'])

#     return loss

