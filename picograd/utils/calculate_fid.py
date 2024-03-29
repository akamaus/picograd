import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def get_moments(x: torch.tensor):
    mean_x = torch.mean(x, 0)
    xm = x - mean_x.expand_as(x)
    c = xm.t().mm(xm)
    c = c / x.size(0)
    return mean_x, c


def calculate_activation_statistics(dataloader, model, classifier):
    classifier.eval()
    model.eval()
    device = next(model.parameters()).device

    # Здесь ожидается что вы пройдете по данным из даталоадера и соберете активации классификатора для реальных и
    # сгенерированных данных
    # После этого посчитаете по ним среднее и ковариацию, по которым посчитаете frechet distance
    # В целом все как в подсчете оригинального FID, но с вашей кастомной моделью классификации
    # note: не забывайте на каком девайсе у вас тензоры
    # note2: не забывайте делать .detach()
    # YOUR CODE
    real_data = []
    gen_data = []
    for image, _ in tqdm(dataloader, desc='computing FID', leave=True):
        image = image.to(device)
        real_data.append(classifier.get_activations(image).detach().cpu())
        gen_data.append(classifier.get_activations(model(image).detach()).detach().cpu())

    real_data = torch.cat(real_data, dim=0)
    gen_data = torch.cat(gen_data, dim=0)
    real_data, gen_data = real_data.view(real_data.shape[0], -1), gen_data.view(gen_data.shape[0], -1)

    real_mean, real_cov = get_moments(real_data)
    gen_mean, gen_cov = get_moments(gen_data)

    return real_mean, real_cov, gen_mean, gen_cov


@torch.no_grad()
def calculate_fid(dataloader, model, classifier):
    m1, s1, m2, s2 = calculate_activation_statistics(dataloader, model, classifier)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value.item()