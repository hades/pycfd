import numpy
import scipy.sparse
import scipy.sparse.linalg
import sys

class Config(object):
    def __init__(self, args=[]):
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
        self.UINLET  = 0.
        self.VINLET  = 0.
        self.WINLET  = 1e-3
        self.URF     = 1.

        for i in args:
            if '=' in i:
                arg, val = i.split('=', 1)
                if not hasattr(self, arg):
                    print "warning: setting new config variable {}".format(arg)
                setattr(self, arg, eval(val, dict((k, getattr(self, k)) for k in dir(self))))
            else:
                print "{} == {}".format(i, getattr(self, i))

def harmonic_average(a, b):
    return 2./(1./a + 1./b)

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
    InternalNode  = numpy.zeros((dim, 1), dtype=numpy.bool)
    UF = numpy.zeros((dim, 1))

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

            this_coeff = harmonic_average(1./my_pcoeff, 1./their_pcoeff) / (config.DX * config.DX)
            PressureMatrix[nodeid, neighb_nodeid] = this_coeff
            my_central_pcoeff += this_coeff

            interp_my = 1./my_pcoeff
            interp_their = 1./their_pcoeff
            wall_interps.append((interp_my / (interp_my + interp_their),
                                 interp_their / (interp_my + interp_their)))

        MomentumMatrix[nodeid, nodeid] = 6. * coeff + my_pcoeff
        PressureMatrix[nodeid, nodeid] = -my_central_pcoeff

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

        UF[nodeid, 0] = 1. / my_pcoeff
        InternalNode[nodeid, 0] = 1

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
            xboundary = boundary
            xinside_direction = inside_direction[:]
            if j == 0:
                boundary = 'wall'
                inside_direction[1] += 1
            if j == config.CELLSY - 1:
                boundary = 'wall'
                inside_direction[1] -= 1

            for k in xrange(config.CELLSZ):
                yboundary = boundary
                yinside_direction = inside_direction[:]
                if k == 0:
                    boundary = 'inlet'
                    inside_direction[2] += 1
                if k == config.CELLSZ - 1:
                    boundary = 'outlet'
                    inside_direction[2] -= 1

                fill_row(i, j, k, boundary, inside_direction)
                boundary = yboundary
                inside_direction = yinside_direction
            boundary = xboundary
            inside_direction = xinside_direction

    return dim, \
           MomentumMatrix.tocsc(), \
           PressureMatrix.tocsc(), \
           DX.tocsc(), \
           DY.tocsc(), \
           DZ.tocsc(), \
           DXg.tocsc(), \
           DYg.tocsc(), \
           DZg.tocsc(), \
           InletVelocity, \
           UF, \
           InternalNode

def solve(matrix, rhs):
    result = scipy.sparse.linalg.spsolve(matrix, rhs)
    if len(result.shape) < 2:
        return result.reshape((result.shape[0], 1))
    return result

def chorin(config):
    print "Creating matrices..."
    dim, MM, PM, DX, DY, DZ, DXg, DYg, DZg, IV, UF, INT = fill_matrices(config)
    print "Problem dimension is {}".format(dim)
    u = numpy.zeros((dim, 1))
    v = numpy.zeros((dim, 1))
    w = numpy.zeros((dim, 1))
    p = numpy.zeros((dim, 1))

    timestep = 0
    time = 0.
    try:
        while True:
            u_old = u.copy()
            v_old = v.copy()
            w_old = w.copy()
            p_old = p.copy()

            dp_dx = DXg * p_old
            dp_dy = DYg * p_old
            dp_dz = DZg * p_old

            u_star = solve(MM, config.UINLET * IV + INT * ((config.RHO / config.DT) * u_old - dp_dx))
            v_star = solve(MM, config.VINLET * IV + INT * ((config.RHO / config.DT) * v_old - dp_dy))
            w_star = solve(MM, config.WINLET * IV + INT * ((config.RHO / config.DT) * w_old - dp_dz))

            div_u_star = DX * u_star + DY * v_star + DZ * w_star

            p_corr = solve(PM, INT * div_u_star)

            dp_corr_dx = DXg * p_corr
            dp_corr_dy = DYg * p_corr
            dp_corr_dz = DZg * p_corr

            u = u_star - config.URF * UF * dp_corr_dx
            v = v_star - config.URF * UF * dp_corr_dy
            w = w_star - config.URF * UF * dp_corr_dz
            p = p + config.URF * p_corr

            print "T=={} ".format(time),
            numpy.savez("T{:04}.npz".format(timestep),
                        dp_dx=dp_dx,
                        dp_dy=dp_dy,
                        dp_dz=dp_dz,
                        u_star=u_star,
                        v_star=v_star,
                        w_star=w_star,
                        div_u_star=div_u_star,
                        p_corr=p_corr,
                        dp_corr_dx=dp_corr_dx,
                        dp_corr_dy=dp_corr_dy,
                        dp_corr_dz=dp_corr_dz,
                        u=u,
                        v=v,
                        w=w,
                        p=p)
            print "umax=={} vmax=={} wmax={} pmax={} ".format(u.max(), v.max(), w.max(), p.max()),
            print
            if time > 100.: break
            if timestep > 500: break
            timestep += 1
            time += config.DT
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    chorin(Config(sys.argv[1:]))
