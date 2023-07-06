import torch

from ConstantList import *
import cv2


class PhantomClass:
    def __init__(self, Dir,
                 Diameter=30e-9,
                 MagSaturation=4.46e5,
                 Concentration=1,
                 Temperature=20.0,
                 RelaxationTime=1e-6,
                 ):

        self._Dir = Dir                         # 路径
        self._Diameter = Diameter               # [m] 粒径
        self._Tt = Temperature + TDT            # [k] 绝对温度
        self._MagSaturation = MagSaturation     # [A/m] 磁饱和强度
        self._Concentration = Concentration     # [mg/mL] [kg/m^3] 浓度
        self._RelaxationTime = RelaxationTime   # [s] 弛豫时间

        self._Volume = self.__get_ParticleVolume()      # [m^3] 体积
        self._ms = self.__get_MagMomentSaturation()     # [A*m^2] 磁矩

        self._Bcoeff = self.__get_ParticleProperty()    # 参数
        self._cParticles = self.__get_Particles()       # [/m^3] 浓度
        self._Picture = self._get_Picture()

    # Get the Volume of the particle.
    def __get_ParticleVolume(self):
        return (self._Diameter ** 3) * PI / 6.0

    # Get the saturation magnetization of the particle.
    def __get_MagMomentSaturation(self):
        return self._MagSaturation * self._Volume

    # Get the property of the particle.
    def __get_ParticleProperty(self):
        return (U0 * self._ms) / (KB * self._Tt)

    # Get the concentration of the particle.
    def __get_Particles(self):
        c_Fe = self._Concentration  # [mg/mL] 溶液浓度
        p_Fe3o4 = 5200              # [Kg/m^3] 四氧化三铁密度
        mol_Fe3O4 = 0.232           # [Kg/mol] Fe3O4摩尔质量
        mol_Fe = 0.056              # [Kg/mol] Fe摩尔质量
        m_core = p_Fe3o4 * self._Volume
        m_Fe = m_core * 3 * mol_Fe / mol_Fe3O4  # 每个核中铁离子质量[Kg]
        c_particles = c_Fe/m_Fe     # 溶液中四氧化三铁浓度[/m^3]
        return c_particles

    def _get_Picture(self, Xn=50, Yn=50):
        img = cv2.imread(self._Dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (Yn, Xn), interpolation=cv2.INTER_CUBIC)
        cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
        img = torch.from_numpy(img)
        return img
