import math
import numpy as np


class AnalyticalSolution:
    def __init__(self, E, nu, Tx, R, offset):
        self.E = E
        self.nu = nu
        self.Tx = Tx
        self.R = R
        self.offset = offset
        self.kappa = (3.0 - nu) / (1.0 + nu)
        self.G = E / (2.0 * (1 + nu))
        self.factor = Tx * R / (8 * self.G)

    def toPolar(self, pos: np.array):
        x = pos[0] - self.offset[0]
        y = pos[1] - self.offset[0]

        r = math.hypot(x, y)
        theta = math.atan2(y, x)

        return r, theta

    def stressSolutionThetaR(self, pos: np.array) -> np.array:
        r, theta = self.toPolar(pos=pos)

        th = self.Tx / 2
        rr2 = math.pow(self.R, 2) / math.pow(r, 2)
        rr4 = math.pow(self.R, 4) / math.pow(r, 4)
        cos2t = math.cos(2 * theta)
        sin2t = math.sin(2 * theta)

        return np.array(
            [
                th * (1 - rr2) + th * (1 + 3 * rr4 - 4 * rr2) * cos2t,
                th * (1 + rr2) - th * (1 + 3 * rr4) * cos2t,
                -th * (1 - 3 * rr4 + 2 * rr2) * sin2t,
            ]
        )

    def coordinateTransform(self, stress: np.array, theta: float) -> np.array:
        cos2t = math.cos(2 * theta)
        sin2t = math.sin(2 * theta)

        s_x = stress[0]
        s_y = stress[1]
        s_xy = stress[2]

        hpl = 0.5 * (s_x + s_y)
        hmi = 0.5 * (s_x - s_y)

        return np.array(
            [
                hpl + hmi * cos2t + s_xy * sin2t,
                hpl - hmi * cos2t - s_xy * sin2t,
                -hmi * sin2t + s_xy * cos2t,
            ]
        )

    def stressSolution(self, pos: np.array) -> np.array:
        _, theta = self.toPolar(pos)
        theta *= -1
        return self.coordinateTransform(self.stressSolutionThetaR(pos), theta)
    
    def displacementSolution(self, pos: np.array) -> np.array:
        r, theta = self.toPolar(pos)

        costh = math.cos(theta)
        sinth = math.sin(theta)
        cos3th = math.cos(3 * theta)
        sin3th = math.cos(3 * theta)

        ra = r/ self.R
        ar = self.R / r
        ar3 = math.pow(self.R, 3) / math.pow(r, 3)

        return np.array([
            self.factor * (ra * (self.kappa + 1) * costh + 2 * ar * ((1 + self.kappa) * costh + cos3th) - 2 * ar3 * cos3th),
            self.factor * (ra * (self.kappa - 3) * sinth + 2 * ar * ((1 - self.kappa) * sinth + sin3th) - 2 * ar3 * sin3th)
        ])

    # Member variables
    E: float
    nu: float
    Tx: float
    R: float
    offset: np.array
    kappa: float
    G: float
    factor: float
