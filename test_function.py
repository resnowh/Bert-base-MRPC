from visualize import visualize_loss


def test_plot_loss():
    train_loss_vals = [0.8, 0.6, 0.5, 0.4, 0.3]
    train_steps = [100, 200, 300, 400, 500]
    eval_loss_vals = [0.7, 0.5, 0.4, 0.3, 0.2]
    eval_steps = [100, 200, 300, 400, 500]

    # 调用 plot_loss 函数进行测试
    visualize_loss(train_loss_vals, train_steps, eval_loss_vals, eval_steps)


# 测试plot_loss()
test_plot_loss()
