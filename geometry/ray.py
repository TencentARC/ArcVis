# -*- coding: utf-8 -*-

import torch

from helper.utils import set_tensor_to_zeros
from .transformation import batch_dot_product


def get_ray_points_by_zvals(rays_o: torch.Tensor, rays_d: torch.Tensor, zvals: torch.Tensor):
    """Get ray points by zvals. Each ray can be sampled by N_pts.
        rays_d is assumed to be normalized.

    Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        zvals: depth values, (N_rays, N_pts)

    Returns:
        rays_pts: final rays points (N_rays, N_pts, 3)
    """
    n_rays = rays_o.shape[0]
    n_pts = zvals.shape[1]
    assert zvals.shape[0] == n_rays, 'Invalid shape for zvals... Should be (N_rays, N_pts)'

    rays_pts = torch.repeat_interleave(rays_o.unsqueeze(1), n_pts, 1)
    rays_pts += torch.einsum('bi, bk->bki', rays_d, zvals)

    return rays_pts


def sphere_ray_intersection(rays_o: torch.Tensor, rays_d: torch.Tensor, radius: torch.Tensor, origin=(0, 0, 0)):
    """Get intersection of ray with sphere surface and the near/far zvals.
    This will be 6 cases: (1)outside no intersection -> near/far: 0, mask = 0
                          (2)outside 1 intersection  -> near = far, mask = 1
                          (3)outside 2 intersections -> near=near>0, far=far
                          (4)inside 1 intersection -> near=0, far=far
                          (5)on surface 1 intersection -> near=0=far=0
                          (6)on surface 2 intersection -> near=0, far=far (tangent/not tangent)
    www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
    Since floating point error exists, we set torch.tensor as 0 for small values, used for tangent case

     Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        radius: sphere radius in (N_r, ) or a single value.
        origin: sphere origin, by default (0, 0, 0). Support only one origin now

    Returns:
        near: near intersection zvals. (N_rays, N_r)
              If only 1 intersection: if not tangent, same as far; else 0. clip by 0.
        far:  far intersection zvals. (N_rays, N_r)
              If only 1 intersection: if not tangent, same as far; else 0.
        pts: (N_rays, N_r, 2, 3), each ray has near/far two points with each sphere.
        mask: (N_rays, N_r), show whether each ray has intersection with the sphere, BoolTensor
     """
    device = rays_o.device
    dtype = rays_o.dtype
    n_rays = rays_o.shape[0]
    # read radius
    if not isinstance(radius, torch.Tensor):
        assert isinstance(radius, float) or isinstance(radius, int), 'Invalid type'
        radius = torch.tensor([radius], dtype=dtype, device=device)
    n_sphere = radius.shape[0]

    rays_o_repeat = torch.repeat_interleave(rays_o, n_sphere, 0)  # (N_rays*N_r, 3)
    rays_d_repeat = torch.repeat_interleave(rays_d, n_sphere, 0)  # (N_rays*N_r, 3)
    r = torch.repeat_interleave(radius.unsqueeze(0), n_rays, 0).view(-1, 1)  # (N_rays*N_r, 3)

    mask = torch.ones(size=(n_rays * n_sphere, 1), dtype=torch.bool, device=device)

    C = torch.tensor([origin], dtype=dtype, device=device)  # (1, 3)
    C = torch.repeat_interleave(C, n_rays * n_sphere, 0)  # (N_rays*N_r, 3)

    OC = C - rays_o_repeat  # (N_rays*N_r, 3)
    z_half = batch_dot_product(OC, rays_d_repeat).unsqueeze(1)  # (N_rays*N_r, 1)
    z_half = set_tensor_to_zeros(z_half)
    rays_o_in_sphere = torch.norm(OC, dim=-1) <= r[:, 0]  # (N_rays*N_r, )
    rays_o_in_sphere = rays_o_in_sphere.unsqueeze(1)  # (N_rays*N_r, 1)
    mask = torch.logical_and(mask, torch.logical_or(z_half > 0, rays_o_in_sphere))  # (N_rays*N_r, 1)

    d_2 = batch_dot_product(OC, OC) - batch_dot_product(z_half, z_half)  # (N_rays*N_r,)
    d_2 = d_2.unsqueeze(1)
    d_2 = set_tensor_to_zeros(d_2)  # (N_rays*N_r, 1)
    mask = torch.logical_and(mask, (d_2 >= 0))  # (N_rays*N_r, 1)

    z_offset = r**2 - d_2  # (N_rays*N_r, 1)
    z_offset = set_tensor_to_zeros(z_offset)
    mask = torch.logical_and(mask, (z_offset >= 0))
    z_offset = torch.sqrt(z_offset)

    near = z_half - z_offset
    near = torch.clamp_min(near, 0.0)
    far = z_half + z_offset
    far = torch.clamp_min(far, 0.0)
    near[~mask], far[~mask] = 0.0, 0.0  # (N_rays*N_r, 1) * 2

    zvals = torch.cat([near, far], dim=1)  # (N_rays*N_r, 2)
    pts = get_ray_points_by_zvals(rays_o_repeat, rays_d_repeat, zvals)  # (N_rays*N_r, 2, 3)

    # reshape
    near = near.contiguous().view(n_rays, n_sphere)
    far = far.contiguous().view(n_rays, n_sphere)
    mask = mask.contiguous().view(n_rays, n_sphere)
    pts = pts.contiguous().view(n_rays, n_sphere, 2, 3)

    return near, far, pts, mask


def aabb_ray_intersection(rays_o: torch.Tensor, rays_d: torch.Tensor, aabb_range: torch.Tensor, eps=1e-7):
    """Get intersection of ray with volume outside surface and the near/far zvals.
    This will be 6 cases: (1)outside no intersection -> near/far: 0, mask = 0
                          (2)outside 1 intersection  -> near = far, mask = 1
                          (3)outside 2 intersections -> near=near>0, far=far (tangent/not tangent)
                          (4)inside 1 intersection -> near=0, far=far
                          (5)on surface 1 intersection -> near=0=far=0
                          (6)on surface 2 intersection -> near=0, far=far (tangent/not tangent)
    www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    Since floating point error exists, we set torch.tensor as 0 for small values, used for tangent case

     Args:
        rays_o: ray origin, (N_rays, 3)
        rays_d: ray direction, assume normalized, (N_rays, 3)
        aabb_range: bbox range of volume, (N_v, 3, 2) of xyz_min/max of each volume
        eps: error threshold for parallel comparison, by default 1e-7

    Returns:
        near: near intersection zvals. (N_rays, N_v)
              If only 1 intersection: if not tangent, same as far; else 0. clip by 0.
        far:  far intersection zvals. (N_rays, N_v)
              If only 1 intersection: if not tangent, same as far; else 0.
        pts: (N_rays, N_v, 2, 3), each ray has near/far two points with each volume.
        mask: (N_rays, N_v), show whether each ray has intersection with the volume, BoolTensor
    """
    device = rays_o.device
    dtype = rays_o.dtype
    n_rays = rays_o.shape[0]
    n_volume = aabb_range.shape[0]
    assert aabb_range.shape[1] == 3 and aabb_range.shape[2] == 2, 'AABB range must be (N, 3, 2)'

    near = torch.zeros((n_rays * n_volume, ), dtype=dtype, device=device)  # (N_rays*N_v,)
    far = torch.ones((n_rays * n_volume, ), dtype=dtype, device=device) * 10000.0  # (N_rays*N_v,)
    aabb_range_repeat = torch.repeat_interleave(aabb_range.unsqueeze(0), n_rays, 0).view(-1, 3, 2)  # (*, 3, 2)
    min_range, max_range = aabb_range_repeat[..., 0], aabb_range_repeat[..., 1]  # (N_rays*N_v, 3)
    mask = torch.ones(size=(n_rays * n_volume, ), dtype=torch.bool, device=device)

    rays_o_repeat = torch.repeat_interleave(rays_o, n_volume, 0)  # (N_rays*N_v, 3)
    rays_d_repeat = torch.repeat_interleave(rays_d, n_volume, 0)  # (N_rays*N_v, 3)

    def update_bound(_rays_o, _rays_d, _min_range, _max_range, _mask, _near, _far, dim=0):
        """Update bound and mask on each dim"""
        _mask_axis = (torch.abs(_rays_d[..., dim]) < eps)  # (N_rays*N_v,)
        _mask_axis_out = torch.logical_or((_rays_o[..., dim] < _min_range[..., dim]),
                                          (_rays_o[..., dim] > _max_range[..., dim]))  # outside the plane
        _mask[torch.logical_and(_mask_axis, _mask_axis_out)] = False

        t1 = (_min_range[..., dim] - _rays_o[..., dim]) / _rays_d[..., dim]
        t2 = (_max_range[..., dim] - _rays_o[..., dim]) / _rays_d[..., dim]
        t = torch.cat([t1[:, None], t2[:, None]], dim=-1)
        t1, _ = torch.min(t, dim=-1)
        t2, _ = torch.max(t, dim=-1)
        update_near = torch.logical_and(_mask, t1 > _near)
        _near[update_near] = t1[update_near]
        update_far = torch.logical_and(_mask, t2 < _far)
        _far[update_far] = t2[update_far]
        _mask[_near > _far] = False

        return _mask, _near, _far

    # x plane
    mask, near, far = update_bound(rays_o_repeat, rays_d_repeat, min_range, max_range, mask, near, far, 0)
    # y plane
    mask, near, far = update_bound(rays_o_repeat, rays_d_repeat, min_range, max_range, mask, near, far, 1)
    # z plane
    mask, near, far = update_bound(rays_o_repeat, rays_d_repeat, min_range, max_range, mask, near, far, 2)

    near, far, mask = near[:, None], far[:, None], mask[:, None]  # (N_rays*N_v, 1)

    near = torch.clamp_min(near, 0.0)
    far = torch.clamp_min(far, 0.0)
    near[~mask], far[~mask] = 0.0, 0.0  # (N_rays*N_v, 1) * 2

    # add some eps for reduce the rounding error
    near[mask] += eps
    far[mask] -= eps

    zvals = torch.cat([near, far], dim=1)  # (N_rays*N_v, 2)
    pts = get_ray_points_by_zvals(rays_o_repeat, rays_d_repeat, zvals)  # (N_rays*N_v, 2, 3)

    # reshape
    near = near.contiguous().view(n_rays, n_volume)
    far = far.contiguous().view(n_rays, n_volume)
    mask = mask.contiguous().view(n_rays, n_volume)
    pts = pts.contiguous().view(n_rays, n_volume, 2, 3)

    return near, far, pts, mask
