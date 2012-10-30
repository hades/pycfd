import numpy
import scipy.sparse
import scipy.sparse.linalg

class Config(object):
    def __init__(self):
        self.RHO     = 847.5
        self.MU      = .17254
        self.K       = 3.17875e-10
        self.DT      = 1.
        self.DX      = 0.05
        self.CELLSX  = 20
        self.CELLSY  = 20
        self.CELLSZ  = 100
        self.POROUS_BEGIN = 40
        self.POROUS_END   = 60

def fill_matrices(config):
    dim = config.CELLSX * config.CELLSY * config.CELLSZ
    MomentumMatrix = scipy.sparse.dok_matrix((dim, dim))
    PressureMatrix = scipy.sparse.dok_matrix((dim, dim))
    DX = scipy.sparse.dok_matrix((dim, dim))
    DY = scipy.sparse.dok_matrix((dim, dim))
    DZ = scipy.sparse.dok_matrix((dim, dim))
    DXg = scipy.sparse.dok_matrix((dim, dim))
    DYg = scipy.sparse.dok_matrix((dim, dim))
    DZg = scipy.sparse.dok_matrix((dim, dim))
    InletVelocity = numpy.zeros((dim, 1))

    def is_porous(i, j, k):
        return k >= config.POROUS_BEGIN and k < config.POROUS_END

    def get_nodeid(i, j, k):
        return i * config.CELLSY * config.CELLSZ \
             + j * config.CELLSZ \
             + k

    def fill_row(i, j, k, boundary, inside_direction):
        nodeid = get_nodeid(i, j, k)
        inside_nodeid = get_nodeid(i + inside_direction[0],
                                   j + inside_direction[1],
                                   k + inside_direction[2])
        if boundary:
            if boundary == 'wall' or boundary == 'inlet':
                MomentumMatrix[nodeid, nodeid] = 1.
                if boundary == 'inlet':
                    InletVelocity[nodeid, 0] = 1.

                PressureMatrix[nodeid, nodeid] = 1.
                PressureMatrix[nodeid, inside_nodeid] = -1.
            elif boundary == 'outlet':
                MomentumMatrix[nodeid, nodeid] = 1.
                MomentumMatrix[nodeid, inside_nodeid] = -1.

                PressureMatrix[nodeid, nodeid] = 1.
            else:
                raise ValueError("unknown boundary {}".format(boundary))
            return

        coeff = config.MU / (config.DX * config.DX)
        muk = 0.
        if is_porous(i, j, k):
            muk = config.MU / config.K
        my_pcoeff = muk + config.RHO / config.DT

        my_central_pcoeff = 0.
        wall_interps = []
        for n in ( (-1, 0, 0),
                   (1, 0, 0),
                   (0, -1, 0),
                   (0, 1, 0),
                   (0, 0, -1),
                   (0, 0, 1), ):
            neighb_nodeid = get_nodeid(i + n[0], j + n[1], k + n[2])
            MomentumMatrix[nodeid, neighb_nodeid] = -coeff

            their_pcoeff = config.RHO / config.DT
            if is_porous(i + n[0], j + n[1], k + n[2]):
                their_pcoeff += config.MU / config.K

            this_coeff = harmonic_average(1./my_pcoeff, 1./their_pcoeff)
            PressureMatrix[nodeid, neighb_nodeid] = this_coeff
            my_central_pcoeff += this_coeff
            wall_interps.append((this_coeff / my_pcoeff, this_coeff / their_pcoeff))

        MomentumMatrix[nodeid, nodeid] = 6. * coeff + my_pcoeff
        PressureMatrix[nodeid, nodeid] = my_central_pcoeff

        DX[nodeid, get_nodeid(i + 1, j, k)] = .5 / config.DX
        DX[nodeid, get_nodeid(i - 1, j, k)] = -.5 / config.DX
        DY[nodeid, get_nodeid(i, j + 1, k)] = .5 / config.DX
        DY[nodeid, get_nodeid(i, j - 1, k)] = -.5 / config.DX
        DZ[nodeid, get_nodeid(i, j, k + 1)] = .5 / config.DX
        DZ[nodeid, get_nodeid(i, j, k - 1)] = -.5 / config.DX

        DXg[nodeid, get_nodeid(i + 1, j, k)] = wall_interps[1][1] / config.DX
        DXg[nodeid, get_nodeid(i - 1, j, k)] = -wall_interps[0][1] / config.DX
        DXg[nodeid, nodeid] = (wall_interps[1][0] - wall_interps[0][0]) / config.DX

        DYg[nodeid, get_nodeid(i, j + 1, k)] = wall_interps[3][1] / config.DX
        DYg[nodeid, get_nodeid(i, j - 1, k)] = -wall_interps[2][1] / config.DX
        DYg[nodeid, nodeid] = (wall_interps[3][0] - wall_interps[2][0]) / config.DX

        DZg[nodeid, get_nodeid(i, j, k + 1)] = wall_interps[5][1] / config.DX
        DZg[nodeid, get_nodeid(i, j, k - 1)] = -wall_interps[4][1] / config.DX
        DZg[nodeid, nodeid] = (wall_interps[5][0] - wall_interps[4][0]) / config.DX

    for i in xrange(config.CELLSX):
        boundary = False
        inside_direction = [0, 0, 0]
        if i == 0:
            boundary = 'wall'
            inside_direction[0] += 1
        if i == config.CELLSX - 1:
            boundary = 'wall'
            inside_direction[0] -= 1

        for j in xrange(config.CELLSY):
            if j == 0:
                boundary = 'wall'
                inside_direction[1] += 1
            if j == config.CELLSY - 1:
                boundary = 'wall'
                inside_direction[1] -= 1

            for k in xrange(config.CELLSZ):
                if k == 0:
                    boundary = 'inlet'
                    inside_direction[2] += 1
                if k == config.CELLSZ - 1:
                    boundary = 'outlet'
                    inside_direction[2] -= 1

                fill_row(i, j, k, boundary, inside_direction)

    return dim, \
           MomentumMatrix.tocsc(), \
           PressureMatrix.tocsc(), \
           DX.tocsc(), \
           DY.tocsc(), \
           DZ.tocsc(), \
           DXg.tocsc(), \
           DYg.tocsc(), \
           DZg.tocsc(), \
           InletVelocity

fill_matrices(Config())
