import HTorch
from HTorch.layers import HEmbedding
import torch
import math
arcosh = HTorch.utils.arcosh
arsinh = HTorch.utils.arsinh

class HyperbolicCone(torch.nn.Module):
    """
    Abstract class to define operations on a hyperbolicCone.
    """
    def __init__(self, source, radius, size, dim, sparse, curvature, margin, sub_apex_dist, energy_type):
        """Initialize HyperbolicCone
        """
        super().__init__()
        self.eps = 1e-15 # this needs to be controlled well, together with HTorch
        self.min_norm = 1e-15
        self.radius = radius
        self.curvature = curvature
        self.margin = margin
        self.sub_apex_dist = sub_apex_dist
        if source == 'origin':
            self.manifold = 'PoincareBall'
            sqrt_c = abs(self.curvature) ** 0.5
            self.level = (torch.tanh(0.5 * sqrt_c * torch.tensor(radius)) + 1e-2) / sqrt_c
        else:
            ### source in ['infinity', float (horocycle light source)]
            self.manifold = 'HalfSpace'
            if source == 'infinity':
                self.level = 1e16
            else:
                self.level = float(source)
        self.emb = HEmbedding(size, dim, sparse=sparse, manifold=self.manifold, curvature=self.curvature)
        self.emb.weight.init_weights(irange=1e-5)
        self.proj_away(self.emb.weight)
        if energy_type == 'angle':
            self.eng = self.angle_energy
        elif energy_type == 'distance':
            self.eng = self.energy
        else:
            return NotImplemented
    
    def proj_away(self, p):
        """inplace handle the case when light source intersects with object"""
        if self.manifold == 'HalfSpace':
            p.data[..., -1].clamp_(max=self.level)
        else:
            """Project a point which intersects with light source to be non-intersect"""
            norm = torch.clamp_min(p.norm(dim=-1, keepdim=True, p=2), self.min_norm)
            proj_p = torch.where(norm < self.level, p / norm * self.level, p).as_subclass(torch.Tensor)
            p.data.copy_(proj_p.data)
            
    def half_aperture(self, p):
        """half_aperture of the cone"""
        raise NotImplemented
        
    def angle(self, parent, children):
        """return angle of children at parent along geodesic with the cone axis"""
        raise NotImplemented
    
    def partial(self, parent, children):
        """check whether children lies in the partial cone defined by parent, i.e., parent > children"""
        angle = self.angle(parent, children)
        half_aperture = self.half_aperture(parent)
        return (angle <= half_aperture)
    
    def dist_to_boundary(self, parent, children):
        """compute distance from children to the boundary of parent cone"""
        raise NotImplemented
    
    def angle_energy(self, parent, children, ConeAxis=False):
        """angle energy to move children inside the parent cone"""
        angle = self.angle(parent, children)
        if ConeAxis:
            ### push towards the central axis of the cone, spoke through p1
            energ = angle
        else:
            ### push towards the (axis+half_aperture) with margin
            half_aperture = self.half_aperture(parent)
            energ = angle - half_aperture
        return energ
    
    def energy(self, parent, children):
        """distance energy to move children inside the parent cone"""
        sqrt_c = abs(self.curvature) ** 0.5
        altitude, dist_to_boundary = self.dist_to_boundary(parent, children)
        if self.training:
            if self.manifold == 'HalfSpace':
                parent = torch.cat([parent[..., :-1], parent[..., -1:] * math.exp(-sqrt_c * self.sub_apex_dist)], dim=-1)
            else:
                norm_parent = torch.clamp_min(parent.norm(dim=-1, p=2, keepdim=True), self.min_norm)
                tmp = (1.0 + sqrt_c * norm_parent) / (1.0 - sqrt_c * norm_parent)
                scale = (math.exp(sqrt_c * self.sub_apex_dist) * tmp - 1.0) / (
                    math.exp(sqrt_c * self.sub_apex_dist) * tmp + 1.0) / (sqrt_c * norm_parent)
                parent = parent * scale
        dist_apex = children.Hdist(parent).squeeze(-1)
        dist = torch.where(altitude>0.0, dist_apex, dist_to_boundary.as_subclass(torch.Tensor))
        return dist
    
    def forward(self, inputs, reverse=False):
        """
        input: 
            contrastive_batch: batch_size * [relation[0]=0, relation[1]=1, 2, 3]
        output: 
            energy between relation[0] and rest: batch_size * (num_pos + num_neg)
        now for training use only
        """
        e = self.emb(inputs)
        with torch.no_grad():
            e.proj_()
        p2 = e.narrow(1, 1, e.size(1) - 1)
        p1 = e.narrow(1, 0, 1).expand_as(p2)
        if reverse:
            parent, children = p2, p1
        else:
            parent, children = p1, p2
        return self.eng(parent, children).squeeze(-1)
    
    def loss(self, energy):
        pos_energy = torch.clamp(energy[..., 0], min=0.0)
        neg_energy = torch.clamp(self.margin - energy[..., 1:], min=0.0).sum(dim=-1)
        return (pos_energy + neg_energy).sum()

    def loss_cross(self, energy):
        """return averaged cross-entropy loss"""
        pos_energy = torch.clamp(energy[..., 0:1], min=0.0)
        neg_energy = torch.clamp(energy[..., 1:]-self.margin, max=0.0)
        exp_neg_energ = torch.exp(-torch.cat([pos_energy, neg_energy],dim=-1))
        prob_pos = exp_neg_energ[..., 0] / torch.sum(exp_neg_energ, axis=-1)
        return - torch.log(prob_pos).mean()
    
class UmbralCone(HyperbolicCone):
    def __init__(self, source, radius, size, dim, sparse=False, curvature=-1.0,
                 margin=0.01, sub_apex_dist=0.01, energy_type='distance'):
        """ Initialize UmbralCone
        """
        super().__init__(source, radius, size, dim, sparse, curvature, margin, sub_apex_dist, energy_type)
    
    def half_aperture(self, p):
        c = abs(self.curvature)
        sqrt_c = c ** 0.5
        if self.manifold == 'HalfSpace':
            base_radius =  math.sinh(sqrt_c * self.radius) * p[..., -1] # may need to handle close boundary situation
            half_aperture = torch.arctan(base_radius / p[..., -1]).as_subclass(torch.Tensor)
        else:
            norm_p = p.norm(dim=-1, p=2)
            sin_beta = math.sinh(sqrt_c * self.radius) * (1.0 - c * norm_p**2)/(2.0 * sqrt_c * norm_p)
            assert torch.all(sin_beta < 1.0), f"Object Ball at this point with radius {self.radius} contains the origin"
            hx = (1.0 / c - norm_p**2) / (2.0 * sin_beta)
            cos_theta = (1.0 / c + norm_p**2) / (2.0 * torch.sqrt(hx**2 + norm_p**2/c))
            half_aperture = torch.pi / 2.0 - torch.arccos(cos_theta)
        return half_aperture
    
    def angle(self, parent, children):
        if self.manifold == 'HalfSpace':
            dist = torch.clamp_min((parent - children).norm(dim=-1, p=2), self.min_norm)
            cos_angle = (parent[..., -1] - children[..., -1]) / dist
            angle_ = torch.arccos(cos_angle).as_subclass(torch.Tensor)
        else:
            c = abs(self.curvature)
            sqrt_c = c ** 0.5
            ### exact version, need to decide sign of beta when combining with alpha, use energy instead for testing
            norm_parent = torch.clamp_min(parent.norm(dim=-1, p=2), self.min_norm)
            sin_beta = math.sinh(sqrt_c * self.radius) * (1.0 - c * norm_parent**2)/(2.0 * sqrt_c * norm_parent)
            beta = torch.asin(sin_beta)
            norm_children = torch.clamp_min(children.norm(dim=-1, p=2), self.min_norm)
            cos_alpha = (torch.sum(parent * children, dim=-1) / (norm_parent * norm_children)).as_subclass(torch.Tensor)
            alpha = torch.acos(cos_alpha)
            h = 0.5 * (norm_parent**2 - norm_children**2) / (norm_children * torch.sin(beta - alpha) - norm_parent * sin_beta)
            R_x_y = norm_parent**2 + h**2 + 2.0 * h * norm_parent * sin_beta
            cos_angle = torch.where(h < R_x_y, h / R_x_y * torch.cos(beta), h / R_x_y * torch.sin(beta))
#             print('cos_angle', cos_angle[torch.abs(cos_angle)>1.0])
#             print('h', h[torch.abs(cos_angle)>1.0])
#             print('R_x_y', R_x_y[torch.abs(cos_angle)>1.0])
#             print('beta', beta[torch.abs(cos_angle)>1.0])
#             print('sin_beta', sin_beta[torch.abs(cos_angle)>1.0])
            assert torch.all(torch.abs(cos_angle)<=1.0), "Getting >1 cos_angle in angle"
#             cos_angle.clamp_(min=-1.0, max=1.0)
            angle_ = torch.arccos(cos_angle).as_subclass(torch.Tensor)
            angle_ = torch.where(alpha>beta, angle_, alpha)
            ### below is just an approximation to compute the angles
#             norm_parent = torch.clamp_min(parent.norm(dim=-1, p=2), self.min_norm)
#             dist = torch.clamp_min((parent - children).norm(dim=-1, p=2), self.min_norm)
#             cos_angle = torch.sum((children - parent) * parent, dim=-1) / (norm_parent * dist)
#             angle_ = torch.arccos(cos_angle).as_subclass(torch.Tensor)
        return angle_
        
    def dist_to_boundary(self, parent, children):
        c = abs(self.curvature)
        sqrt_c = c ** 0.5
        if self.manifold == 'HalfSpace':
            temperature = self.temperature(parent, children)
            dist_to_boundary = arsinh(temperature / children[..., -1]) / sqrt_c + self.radius
            altitude = self.altitude(parent, children, temperature)
        else:
            ## compute temperature
            norm_parent = torch.clamp_min(parent.norm(dim=-1, p=2), self.min_norm)
            sin_beta = math.sinh(sqrt_c * self.radius) * (1.0 - c * norm_parent**2)/(2.0 * sqrt_c * norm_parent)
            beta = torch.asin(sin_beta)
            norm_children = torch.clamp_min(children.norm(dim=-1, p=2), self.min_norm)
            cos_alpha = (torch.sum(parent * children, dim=-1) / (norm_parent * norm_children)).as_subclass(torch.Tensor)
            alpha = torch.acos(cos_alpha)
            theta = alpha - beta
            temperature = 2.0 * sqrt_c * norm_children * torch.sin(theta)
            ## compute altitude
            height_children = (1.0 + c*norm_children**2) / torch.sqrt((1.0 - c*norm_children**2)**2 + temperature**2)
            height_parent = arcosh(math.cosh(sqrt_c * self.radius) * (1.0 - c*norm_parent**2)/(1.0 + c*norm_parent**2))/sqrt_c
            altitude = height_parent - height_children
            dist_to_boundary = arsinh(temperature / (1.0 - c*norm_children**2)) / sqrt_c + self.radius
        return altitude, dist_to_boundary
    
    def temperature(self, parent, children):
        c = abs(self.curvature)
        sqrt_c = c ** 0.5
        if self.manifold == 'HalfSpace':
            dist = (parent[..., :-1] - children[..., :-1]).norm(dim=-1, p=2)
            temperature = (dist - math.sinh(sqrt_c * self.radius) * parent[..., -1]).as_subclass(torch.Tensor)
        else:
            norm_parent = torch.clamp_min(parent.norm(dim=-1, p=2), self.min_norm)
            sin_beta = math.sinh(sqrt_c * self.radius) * (1.0 - c * norm_parent**2)/(2.0 * sqrt_c * norm_parent)
            beta = torch.asin(sin_beta)
            norm_children = torch.clamp_min(children.norm(dim=-1, p=2), self.min_norm)
            cos_alpha = (torch.sum(parent * children, dim=-1) / (norm_parent * norm_children)).as_subclass(torch.Tensor)
            theta = alpha - beta
            temperature = 2.0 * sqrt_c * norm_children * torch.sin(theta)
        return temperature

    def altitude(self, parent, children, temperature):
        c = abs(self.curvature)
        sqrt_c = c ** 0.5
        if self.manifold == 'HalfSpace':
            height =  math.cosh(sqrt_c * self.radius) * parent[..., -1]
            altitude = (temperature**2 + children[..., -1]**2 - height**2).as_subclass(torch.Tensor)
        else:
            norm_parent = torch.clamp_min(parent.norm(dim=-1, p=2), self.min_norm)
            norm_children = torch.clamp_min(children.norm(dim=-1, p=2), self.min_norm)
            height_children = (1.0 + c*norm_children**2) / torch.sqrt((1.0 - c*norm_children**2)**2 + temperature**2)
            height_parent = arcosh(math.cosh(sqrt_c * self.radius) * (1.0 - c*norm_parent**2)/(1.0 + c*norm_parent**2))/sqrt_c
            altitude = height_parent - height_children
        return altitude
    
class PeumbralCone(HyperbolicCone):
    def __init__(self, source, radius, size, dim, sparse=False, curvature=-1.0, 
                 margin=0.1, sub_apex_dist=0.01, energy_type='distance'):
        """ Initialize PeumbralCone
        """
        super().__init__(source, radius, size, dim, sparse, curvature, margin, sub_apex_dist, energy_type)
    
    def half_aperture(self, p):
        if self.manifold == 'HalfSpace':
            sin_theta = p[..., -1] / self.level
        else:
            c = abs(self.curvature)
            sqrt_c = c ** 0.5
            norm_p = p.norm(dim=-1, p=2)
            sin_theta = math.sinh(sqrt_c * self.radius) * (1.0 - c * norm_p**2)/(2.0 * sqrt_c * norm_p)
        half_aperture = sin_theta.clamp_(min=-1.0, max=1.0).asin().as_subclass(torch.Tensor)
        return half_aperture
    
    def angle(self, parent, children):
        c = abs(self.curvature)
        if self.manifold == 'HalfSpace':
            log_p_c = parent.logmap(parent, children)
            vertical_u = torch.zeros(log_p_c.size(-1)).to(log_p_c)
            vertical_u[-1] = -1.0
            norm = torch.clamp_min(log_p_c.norm(dim=-1, p=2), self.min_norm)
            cos_angle = torch.sum(vertical_u * log_p_c, dim=-1) / norm
            angle_ = torch.arccos(cos_angle)
        else:
            angle_ = self.emb.weight.manifold.angle_at_x(parent, children, c)
        return angle_
        
    def dist_to_boundary(self, parent, children):
        sqrt_c = abs(self.curvature) ** 0.5
        alpha = self.angle(parent, children)
        half_aperture = self.half_aperture(parent)
        theta = alpha - half_aperture
        dist_apex_parent = children.Hdist(parent).squeeze(-1)
        dist_to_boundary = arsinh(torch.sinh(sqrt_c * dist_apex_parent) * torch.sin(theta)) / sqrt_c
        altitude = theta - torch.pi/2.0
        return altitude, dist_to_boundary