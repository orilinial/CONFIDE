import numpy as np
import torch
import pde
import matplotlib.pyplot as plt


def test_params(pred_params, gt_params, epoch, pde_type):
    if pde_type == 'fn2d_u':
        fig, axs = plt.subplots(2, 2)
        fig.suptitle(f'Epoch {epoch} -- Function estimation - Ru, Rv')
        axs[0, 0].set_title('GT')
        axs[0, 1].set_title('Pred')
        axs[0, 0].set_ylabel('Ru')
        axs[1, 0].set_ylabel('Rv')

        v_min = gt_params[50, :, :, 0].min()
        v_max = gt_params[50, :, :, 0].max()
        axs[0, 0].imshow(gt_params[50, :, :, 0], vmin=v_min, vmax=v_max)
        im = axs[0, 1].imshow(pred_params[50, :, :, 0], vmin=v_min, vmax=v_max)
        fig.colorbar(im, ax=axs[0, :])

        v_min = gt_params[50, :, :, 1].min()
        v_max = gt_params[50, :, :, 1].max()
        axs[1, 0].imshow(gt_params[50, :, :, 1], vmin=v_min, vmax=v_max)
        im = axs[1, 1].imshow(pred_params[50, :, :, 1], vmin=v_min, vmax=v_max)
        fig.colorbar(im, ax=axs[1, :])

    elif pde_type == 'fn2d':
        fig, axs = plt.subplots(1, 3)
        fig.suptitle(f'Epoch {epoch} -- Function estimation - k, Rv')

        axs[0].scatter(gt_params[:, 0, 0, 0], pred_params[:, 0, 0, 0])
        axs[0].set_xlabel('GT')
        axs[0].set_ylabel('Pred')

        axs[1].set_title('GT-Rv')
        axs[2].set_title('Pred-Rv')

        v_min = gt_params[50, :, :].min()
        v_max = gt_params[50, :, :].max()
        axs[1].imshow(gt_params[50, :, :, 1], vmin=v_min, vmax=v_max)
        im = axs[2].imshow(pred_params[50, :, :, 1], vmin=v_min, vmax=v_max)

    plt.show()


def test_model(delta_t, delta_x, t_len, x_len, model, context, gt_sol, gt_params, epoch, pde_type, save_vid=False):
    if pde_type == 'fn2d_u':
        known = False
    else:
        known = True

    pred_sol, _ = solve_pde(
        delta_t=delta_t,
        delta_x=delta_x,
        t_len=t_len,
        x_len=x_len,
        model=model,
        context=context,
        known=known)

    ts = [0, 20, 40, 60, 80]

    # Demonstrate U
    fig, axs = plt.subplots(2, len(ts))
    fig.suptitle(f'Epoch {epoch}; U')
    v_min = gt_sol[:, 0].min()
    v_max = gt_sol[:, 0].max()

    for i, t in enumerate(ts):
        axs[0, i].imshow(gt_sol[t, 0], vmin=v_min, vmax=v_max)
        axs[0, i].set_title(f't={t}', fontsize=12)
        axs[1, i].imshow(pred_sol[t, 0], vmin=v_min, vmax=v_max)
        axs[1, i].set_title(f't={t}', fontsize=12)

    axs[0, 0].set_ylabel('GT', fontsize=15)
    axs[1, 0].set_ylabel('Pred', fontsize=15)

    # Demonstrate V
    fig, axs = plt.subplots(2, len(ts))
    fig.suptitle(f'Epoch {epoch}; V')
    v_min = gt_sol[:, 0].min()
    v_max = gt_sol[:, 0].max()

    for i, t in enumerate(ts):
        axs[0, i].imshow(gt_sol[t, 1], vmin=v_min, vmax=v_max)
        axs[0, i].set_title(f't={t}', fontsize=12)
        axs[1, i].imshow(pred_sol[t, 1], vmin=v_min, vmax=v_max)
        axs[1, i].set_title(f't={t}', fontsize=12)

    axs[0, 0].set_ylabel('GT', fontsize=15)
    axs[1, 0].set_ylabel('Pred', fontsize=15)

    if save_vid:
        import imageio
        images_gt_u = []
        images_gt_v = []
        images_pred_u = []
        images_pred_v = []
        for t in range(pred_sol.shape[0] - 1):
            images_gt_u.append(gt_sol[t, 0])
            images_gt_v.append(gt_sol[t, 1])
            images_pred_u.append(pred_sol[t, 0])
            images_pred_v.append(pred_sol[t, 1])

        imageio.mimsave(f'results/fn2d/images_gt_u_{epoch}.gif', images_gt_u)
        imageio.mimsave(f'results/fn2d/images_gt_v_{epoch}.gif', images_gt_v)
        imageio.mimsave(f'results/fn2d/images_pred_u_{epoch}.gif', images_pred_u)
        imageio.mimsave(f'results/fn2d/images_pred_v_{epoch}.gif', images_pred_v)
        # quit()


def solve_pde(delta_t, delta_x, t_len, x_len, model, context, known):
    x_low = -x_len
    y_low = -x_len
    x_high = x_len
    y_high = x_len

    delta_y = delta_x
    x_len = x_high - x_low
    y_len = y_high - y_low
    grid = pde.CartesianGrid([(x_low, x_high), (y_low, y_high)], [x_len // delta_x, y_len // delta_y])

    state = pde.FieldCollection([
        pde.ScalarField(grid, data=context[0, 0]),
        pde.ScalarField(grid, data=context[0, 1])
    ])

    if known:
        eq = FN2DPDEKnown(grid.axes_coords[0], model, context)
    else:
        eq = FN2DPDEUnknown(grid.axes_coords[0], model, context)

    storage = pde.MemoryStorage()
    eq.solve(state, t_range=t_len, dt=delta_t / 10.0, tracker=storage.tracker(delta_t))

    times = storage.times
    t = torch.FloatTensor(times[:-1]).unsqueeze(1)
    f = torch.FloatTensor(np.array(storage.data[:-1]))
    _, parameters = model.forward_multiple_t(t, f, torch.FloatTensor(context).unsqueeze(0))
    parameters = parameters.squeeze().numpy()

    return np.array(storage.data), parameters


class FN2DPDEUnknown(pde.PDEBase):
    def __init__(self, x, model, context):
        super(FN2DPDEUnknown, self).__init__()
        self.bc = "auto_periodic_neumann"
        self.x = torch.FloatTensor(x).unsqueeze(1)
        self.context = torch.FloatTensor(context).unsqueeze(0)
        self.model = model

    def evolution_rate(self, state, t=0):
        u, v = state
        t_tensor = torch.FloatTensor([t]).unsqueeze(0)
        state_tensor = torch.FloatTensor(state.data).unsqueeze(0)
        _, parameters = self.model(t_tensor, state_tensor, self.context)
        Ru = parameters[..., 0].squeeze().numpy()
        Rv = parameters[..., 1].squeeze().numpy()
        a = 1e-3
        b = 5e-3

        du_dt = a * u.laplace(bc=self.bc) + Ru
        dv_dt = b * v.laplace(bc=self.bc) + Rv
        return pde.FieldCollection([du_dt, dv_dt])

class FN2DPDEKnown(pde.PDEBase):
    def __init__(self, x, model, context):
        super(FN2DPDEKnown, self).__init__()
        self.bc = "auto_periodic_neumann"
        self.x = torch.FloatTensor(x).unsqueeze(1)
        self.context = torch.FloatTensor(context).unsqueeze(0)
        self.model = model

    def evolution_rate(self, state, t=0):
        u, v = state
        t_tensor = torch.FloatTensor([t]).unsqueeze(0)
        state_tensor = torch.FloatTensor(state.data).unsqueeze(0)
        _, parameters = self.model(t_tensor, state_tensor, self.context)

        k = parameters[..., 0].squeeze().numpy()
        Ru = u - u ** 3 - k - v
        Rv = parameters[..., 1].squeeze().numpy()

        a = 1e-3
        b = 5e-3

        du_dt = a * u.laplace(bc=self.bc) + Ru
        dv_dt = b * v.laplace(bc=self.bc) + Rv
        return pde.FieldCollection([du_dt, dv_dt])
