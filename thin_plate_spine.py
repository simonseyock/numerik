import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import pyplot


def g(r):
    r = np.maximum(r, 1e-08)
    return r**2 * np.log(r)


def norm_2(x):
    return np.sqrt(np.sum(np.abs(x) ** 2))


def compute_tps_weights(xs, ys, zs):
    """This function constructs a thin plate spline interpolating the given three-
           dimensional points by means of a two-dimensional function.
          \param X,Y,Z Three arrays of shape (n,) containing n points in three-
                 dimensional space.
          \return An array of shape identical to X where the i-th entry stores the
                  weight to be used for the radial basis function centered at
                  (X[i],Y[i])."""

    n = xs.shape[0]
    a = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            a[i][j] = g(norm_2([xs[i] - xs[j], ys[i] - ys[j]]))

    return np.linalg.solve(a, zs)


def evaluate_tps_spine(xs_new, ys_new, xs, ys, weights):
    """Given the weights for a thin plate spline as returned by ComputeTPSWeights
       this function evaluates the thin plate spline at prescribed locations.
      \param XNew,YNew The x and y coordinates at which the TPS spline should be
             evaluated. These are np.ndarray objects of arbitrary but identical
             shape.
      \param X,Y The coordinates passed to ComputeTPSWeights().
      \param Weights The weights returned by ComputeTPSWeights().
      \return An array of shape identical to XNew containing the value of the thin
              plate spline at the coordinates given by XNew and YNew."""

    zs_new = np.zeros(xs_new.shape)

    for index in np.ndindex(xs_new.shape):
        for k in range(weights.shape[0]):
            zs_new[index] += weights[k] * g(norm_2([xs_new[index] - xs[k], ys_new[index] - ys[k]]))

    return zs_new


if (__name__ == "__main__"):
    # Produce random points which are to be interpolated by the thin plate spline
    nPoint = 20
    X = np.random.rand(nPoint)
    Y = np.random.rand(nPoint)
    Z = np.random.rand(nPoint)
    # Produce a regular grid for evaluation of the thin plate spline
    nGridCell = 41
    XNew = np.linspace(0.0, 1.0, nGridCell)
    YNew = np.linspace(0.0, 1.0, nGridCell)
    XNew, YNew = np.meshgrid(XNew, YNew)
    # Construct and evaluate the thin plate spline
    Weights = compute_tps_weights(X, Y, Z)
    ZNew = evaluate_tps_spine(XNew, YNew, X, Y, Weights)

    # Check whether all points have been fitted correctly
    ZReconstructed = evaluate_tps_spine(X, Y, X, Y, Weights)
    print("If the following number is nearly zero, the solution appears to be working fine.")
    print(np.linalg.norm(Z - ZReconstructed))

    # Plot the input points and the interpolated function
    Axes = pyplot.subplot(projection="3d")
    Axes.scatter3D(X, Y, Z, color="r")
    Axes.set_xlim(0.0, 1.0)
    Axes.set_ylim(0.0, 1.0)
    Axes.set_zlim(np.min(ZNew), np.max(ZNew))
    Axes.plot_wireframe(XNew, YNew, ZNew, rstride=1, cstride=1)
    pyplot.show()
