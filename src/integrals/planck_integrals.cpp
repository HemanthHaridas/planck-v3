#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <array>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace Hermite {

struct Primitive {
    double primitive_exp = 0.0;
    double orbital_coeff = 1.0;
    double orbital_norm = 1.0;
};

struct Contracted {
    double location_x = 0.0;
    double location_y = 0.0;
    double location_z = 0.0;

    std::int64_t shell_x = 0;
    std::int64_t shell_y = 0;
    std::int64_t shell_z = 0;

    std::vector<Primitive> contracted_GTO;
};

struct GaussianProduct {
    double combined_exp = 0.0;
    double product_center_x = 0.0;
    double product_center_y = 0.0;
    double product_center_z = 0.0;
};

static inline GaussianProduct computeGaussianProduct(
    const Primitive& primitiveA,
    double xA,
    double yA,
    double zA,
    const Primitive& primitiveB,
    double xB,
    double yB,
    double zB
) {
    const double a = primitiveA.primitive_exp;
    const double b = primitiveB.primitive_exp;
    const double p = a + b;
    if (p <= 0.0) {
        throw std::invalid_argument("Primitive exponents must be positive.");
    }

    GaussianProduct gp;
    gp.combined_exp = p;
    gp.product_center_x = (a * xA + b * xB) / p;
    gp.product_center_y = (a * yA + b * yB) / p;
    gp.product_center_z = (a * zA + b * zB) / p;
    return gp;
}

struct Overlap {
    /// Computes the Hermite overlap integral in one dimension.
    static double computePrimitive1D(
        double exponentA,
        double centerA,
        std::int64_t shellA,
        double exponentB,
        double centerB,
        std::int64_t shellB,
        std::int64_t hermiteNodes
    ) {
        if (shellA < 0 || shellB < 0) {
            return 0.0;
        }

        const double combExp = exponentA + exponentB;
        if (combExp <= 0.0) {
            throw std::invalid_argument("Primitive exponents must be positive.");
        }
        const double gaussExp = (exponentA * exponentB) / combExp;

        if (hermiteNodes < 0 || hermiteNodes > (shellA + shellB)) {
            return 0.0;
        }

        if (shellA == 0 && shellB == 0 && hermiteNodes == 0) {
            const double dx = centerA - centerB;
            return std::exp(-gaussExp * dx * dx);
        }

        if (shellB == 0) {
            return (
                (1.0 / (2.0 * combExp)) *
                    Overlap::computePrimitive1D(exponentA, centerA, shellA - 1, exponentB, centerB, shellB, hermiteNodes - 1)
                - (exponentB * (centerA - centerB) / combExp) *
                    Overlap::computePrimitive1D(exponentA, centerA, shellA - 1, exponentB, centerB, shellB, hermiteNodes)
                + (hermiteNodes + 1) *
                    Overlap::computePrimitive1D(exponentA, centerA, shellA - 1, exponentB, centerB, shellB, hermiteNodes + 1)
            );
        }

        return (
            (1.0 / (2.0 * combExp)) *
                Overlap::computePrimitive1D(exponentA, centerA, shellA, exponentB, centerB, shellB - 1, hermiteNodes - 1)
            + (exponentA * (centerA - centerB) / combExp) *
                Overlap::computePrimitive1D(exponentA, centerA, shellA, exponentB, centerB, shellB - 1, hermiteNodes)
            + (hermiteNodes + 1) *
                Overlap::computePrimitive1D(exponentA, centerA, shellA, exponentB, centerB, shellB - 1, hermiteNodes + 1)
        );
    }

    /// Computes the three-dimensional Hermite overlap integral.
    static double computePrimitive3D(
        const Primitive& primitiveA,
        double xA,
        double yA,
        double zA,
        std::int64_t lxA,
        std::int64_t lyA,
        std::int64_t lzA,
        const Primitive& primitiveB,
        double xB,
        double yB,
        double zB,
        std::int64_t lxB,
        std::int64_t lyB,
        std::int64_t lzB
    ) {
        const double ax = Overlap::computePrimitive1D(primitiveA.primitive_exp, xA, lxA, primitiveB.primitive_exp, xB, lxB, 0);
        const double ay = Overlap::computePrimitive1D(primitiveA.primitive_exp, yA, lyA, primitiveB.primitive_exp, yB, lyB, 0);
        const double az = Overlap::computePrimitive1D(primitiveA.primitive_exp, zA, lzA, primitiveB.primitive_exp, zB, lzB, 0);

        const double combExp = primitiveA.primitive_exp + primitiveB.primitive_exp;
        if (combExp <= 0.0) {
            throw std::invalid_argument("Primitive exponents must be positive.");
        }

        constexpr double pi = 3.141592653589793238462643383279502884;
        return ax * ay * az * std::pow(pi / combExp, 1.5);
    }

    /// Computes the contracted overlap integral between two contracted Gaussian functions.
    static double computeContracted(const Contracted& contractedGaussianA, const Contracted& contractedGaussianB) {
        const double xA = contractedGaussianA.location_x;
        const double yA = contractedGaussianA.location_y;
        const double zA = contractedGaussianA.location_z;

        const std::int64_t lxA = contractedGaussianA.shell_x;
        const std::int64_t lyA = contractedGaussianA.shell_y;
        const std::int64_t lzA = contractedGaussianA.shell_z;

        const double xB = contractedGaussianB.location_x;
        const double yB = contractedGaussianB.location_y;
        const double zB = contractedGaussianB.location_z;

        const std::int64_t lxB = contractedGaussianB.shell_x;
        const std::int64_t lyB = contractedGaussianB.shell_y;
        const std::int64_t lzB = contractedGaussianB.shell_z;

        double integral = 0.0;

        const std::size_t nA = contractedGaussianA.contracted_GTO.size();
        const std::size_t nB = contractedGaussianB.contracted_GTO.size();

        for (std::size_t ii = 0; ii < nA; ii++) {
            for (std::size_t jj = 0; jj < nB; jj++) {
                [[maybe_unused]] auto productGaussianAB =
                    computeGaussianProduct(contractedGaussianA.contracted_GTO[ii], xA, yA, zA, contractedGaussianB.contracted_GTO[jj], xB, yB, zB);

                double value = Overlap::computePrimitive3D(
                    contractedGaussianA.contracted_GTO[ii], xA, yA, zA, lxA, lyA, lzA,
                    contractedGaussianB.contracted_GTO[jj], xB, yB, zB, lxB, lyB, lzB);

                value *= contractedGaussianA.contracted_GTO[ii].orbital_coeff * contractedGaussianA.contracted_GTO[ii].orbital_norm;
                value *= contractedGaussianB.contracted_GTO[jj].orbital_coeff * contractedGaussianB.contracted_GTO[jj].orbital_norm;
                integral += value;
            }
        }

        return integral;
    }
};

struct Kinetic {
    static double computePrimitive3D(
        const Primitive& primitiveA,
        double xA,
        double yA,
        double zA,
        std::int64_t lxA,
        std::int64_t lyA,
        std::int64_t lzA,
        const Primitive& primitiveB,
        double xB,
        double yB,
        double zB,
        std::int64_t lxB,
        std::int64_t lyB,
        std::int64_t lzB
    ) {
        double integral = Overlap::computePrimitive3D(primitiveA, xA, yA, zA, lxA, lyA, lzA, primitiveB, xB, yB, zB, lxB, lyB, lzB);
        integral *= (primitiveB.primitive_exp * (2.0 * (lxB + lyB + lzB) + 3.0));

        integral -= (2.0 * std::pow(primitiveB.primitive_exp, 2.0)) *
            Overlap::computePrimitive3D(primitiveA, xA, yA, zA, lxA, lyA, lzA, primitiveB, xB, yB, zB, lxB + 2, lyB, lzB);
        integral -= (2.0 * std::pow(primitiveB.primitive_exp, 2.0)) *
            Overlap::computePrimitive3D(primitiveA, xA, yA, zA, lxA, lyA, lzA, primitiveB, xB, yB, zB, lxB, lyB + 2, lzB);
        integral -= (2.0 * std::pow(primitiveB.primitive_exp, 2.0)) *
            Overlap::computePrimitive3D(primitiveA, xA, yA, zA, lxA, lyA, lzA, primitiveB, xB, yB, zB, lxB, lyB, lzB + 2);

        integral -= (0.5 * lxB * (lxB - 1)) *
            Overlap::computePrimitive3D(primitiveA, xA, yA, zA, lxA, lyA, lzA, primitiveB, xB, yB, zB, lxB - 2, lyB, lzB);
        integral -= (0.5 * lyB * (lyB - 1)) *
            Overlap::computePrimitive3D(primitiveA, xA, yA, zA, lxA, lyA, lzA, primitiveB, xB, yB, zB, lxB, lyB - 2, lzB);
        integral -= (0.5 * lzB * (lzB - 1)) *
            Overlap::computePrimitive3D(primitiveA, xA, yA, zA, lxA, lyA, lzA, primitiveB, xB, yB, zB, lxB, lyB, lzB - 2);

        return integral;
    }

    static double computeContracted(const Contracted& contractedGaussianA, const Contracted& contractedGaussianB) {
        const double xA = contractedGaussianA.location_x;
        const double yA = contractedGaussianA.location_y;
        const double zA = contractedGaussianA.location_z;

        const std::int64_t lxA = contractedGaussianA.shell_x;
        const std::int64_t lyA = contractedGaussianA.shell_y;
        const std::int64_t lzA = contractedGaussianA.shell_z;

        const double xB = contractedGaussianB.location_x;
        const double yB = contractedGaussianB.location_y;
        const double zB = contractedGaussianB.location_z;

        const std::int64_t lxB = contractedGaussianB.shell_x;
        const std::int64_t lyB = contractedGaussianB.shell_y;
        const std::int64_t lzB = contractedGaussianB.shell_z;

        double integral = 0.0;

        const std::size_t nA = contractedGaussianA.contracted_GTO.size();
        const std::size_t nB = contractedGaussianB.contracted_GTO.size();

        for (std::size_t ii = 0; ii < nA; ii++) {
            for (std::size_t jj = 0; jj < nB; jj++) {
                [[maybe_unused]] auto productGaussianAB =
                    computeGaussianProduct(contractedGaussianA.contracted_GTO[ii], xA, yA, zA, contractedGaussianB.contracted_GTO[jj], xB, yB, zB);

                double value = Kinetic::computePrimitive3D(
                    contractedGaussianA.contracted_GTO[ii], xA, yA, zA, lxA, lyA, lzA,
                    contractedGaussianB.contracted_GTO[jj], xB, yB, zB, lxB, lyB, lzB);

                value *= contractedGaussianA.contracted_GTO[ii].orbital_coeff * contractedGaussianA.contracted_GTO[ii].orbital_norm;
                value *= contractedGaussianB.contracted_GTO[jj].orbital_coeff * contractedGaussianB.contracted_GTO[jj].orbital_norm;
                integral += value;
            }
        }

        return integral;
    }
};

} // namespace Hermite

// Helper functions for Huzinaga integrals
namespace {
    /// Compute binomial coefficient C(n, k)
    inline double combination(std::int64_t n, std::int64_t k) {
        if (k < 0 || k > n) {
            return 0.0;
        }
        if (k == 0 || k == n) {
            return 1.0;
        }
        if (k > n - k) {
            k = n - k; // Take advantage of symmetry
        }
        double result = 1.0;
        for (std::int64_t i = 0; i < k; ++i) {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }

    /// Compute double factorial (2n-1)!!
    inline double doublefactorial(std::int64_t n) {
        if (n < 0) {
            return 1.0; // Convention: (-1)!! = 1
        }
        if (n == 0 || n == 1) {
            return 1.0;
        }
        double result = 1.0;
        for (std::int64_t i = n; i > 0; i -= 2) {
            result *= static_cast<double>(i);
        }
        return result;
    }
}

namespace Huzinaga {

struct HuzinagaGaussianProduct {
    std::array<double, 3> gaussian_center = {0.0, 0.0, 0.0};
    std::array<double, 4> gaussian_integral = {0.0, 0.0, 0.0, 0.0};
    double combined_exp = 0.0;
};

static inline HuzinagaGaussianProduct computeGaussianProduct(
    const Hermite::Primitive& primitiveA,
    double xA,
    double yA,
    double zA,
    const Hermite::Primitive& primitiveB,
    double xB,
    double yB,
    double zB
) {
    const double a = primitiveA.primitive_exp;
    const double b = primitiveB.primitive_exp;
    const double p = a + b;
    if (p <= 0.0) {
        throw std::invalid_argument("Primitive exponents must be positive.");
    }

    HuzinagaGaussianProduct gp;
    gp.combined_exp = p;
    gp.gaussian_center[0] = (a * xA + b * xB) / p;
    gp.gaussian_center[1] = (a * yA + b * yB) / p;
    gp.gaussian_center[2] = (a * zA + b * zB) / p;

    // Compute Gaussian integral prefactor: exp(-(a*b/(a+b)) * R_AB^2)
    const double dx = xA - xB;
    const double dy = yA - yB;
    const double dz = zA - zB;
    const double R_AB_sq = dx * dx + dy * dy + dz * dz;
    const double prefactor = std::exp(-(a * b / p) * R_AB_sq);

    constexpr double pi = 3.141592653589793238462643383279502884;
    const double pi_over_p = pi / p;
    
    // Store integral values for different powers
    gp.gaussian_integral[0] = std::pow(pi_over_p, 0.5) * prefactor;
    gp.gaussian_integral[1] = std::pow(pi_over_p, 1.0) * prefactor;
    gp.gaussian_integral[2] = std::pow(pi_over_p, 1.5) * prefactor;
    gp.gaussian_integral[3] = std::pow(pi_over_p, 1.5) * prefactor; // Used in 3D integrals

    return gp;
}

struct Overlap {
    static double expansionIndex1(
        std::int64_t expIndex,
        std::int64_t shellA,
        std::int64_t shellB,
        double distPA,
        double distPB
    ) {
        std::int64_t cMax = (expIndex <= shellA) ? expIndex : shellA;
        std::int64_t cMin = (0 > (expIndex - shellB)) ? 0 : (expIndex - shellB);
        double expansionCoeff = 0.0;

        for (std::int64_t ii = cMin; ii <= cMax; ii++) {
            double aux = combination(shellA, ii);
            aux *= combination(shellB, expIndex - ii);
            aux *= std::pow(distPA, shellA - ii);
            aux *= std::pow(distPB, shellB + ii - expIndex);
            expansionCoeff += aux;
        }
        return expansionCoeff;
    }

    static double computePrimitive1D(
        double exponentA,
        double centerA,
        std::int64_t shellA,
        double exponentB,
        double centerB,
        std::int64_t shellB,
        double gaussianCenter
    ) {
        if (shellA < 0 || shellB < 0) {
            return 0.0;
        }

        const double combExp = exponentA + exponentB;
        if (combExp <= 0.0) {
            throw std::invalid_argument("Primitive exponents must be positive.");
        }

        const double gaussian = ((exponentA * centerA) + (exponentB * centerB)) / combExp;
        double integral = 0.0;

        const std::int64_t max_ii = (shellA + shellB) / 2;
        for (std::int64_t ii = 0; ii <= max_ii; ii++) {
            double value = doublefactorial((2 * ii) - 1);
            value /= std::pow(2.0 * combExp, ii);
            value *= expansionIndex1(2 * ii, shellA, shellB, gaussian - centerA, gaussian - centerB);
            integral += value;
        }

        return integral;
    }

    static double computePrimitive3D(
        const Hermite::Primitive& primitiveA,
        double xA,
        double yA,
        double zA,
        std::int64_t lxA,
        std::int64_t lyA,
        std::int64_t lzA,
        const Hermite::Primitive& primitiveB,
        double xB,
        double yB,
        double zB,
        std::int64_t lxB,
        std::int64_t lyB,
        std::int64_t lzB,
        const std::array<double, 3>& gaussianCenter,
        double gaussianIntegral
    ) {
        const double xDir = computePrimitive1D(
            primitiveA.primitive_exp, xA, lxA,
            primitiveB.primitive_exp, xB, lxB,
            gaussianCenter[0]
        );
        const double yDir = computePrimitive1D(
            primitiveA.primitive_exp, yA, lyA,
            primitiveB.primitive_exp, yB, lyB,
            gaussianCenter[1]
        );
        const double zDir = computePrimitive1D(
            primitiveA.primitive_exp, zA, lzA,
            primitiveB.primitive_exp, zB, lzB,
            gaussianCenter[2]
        );

        constexpr double pi = 3.141592653589793238462643383279502884;
        const double combExp = primitiveA.primitive_exp + primitiveB.primitive_exp;
        return xDir * yDir * zDir * std::pow(pi / combExp, 1.5) * gaussianIntegral;
    }

    static double computeContracted(
        const Hermite::Contracted& contractedGaussianA,
        const Hermite::Contracted& contractedGaussianB
    ) {
        double integral = 0.0;

        const std::size_t nA = contractedGaussianA.contracted_GTO.size();
        const std::size_t nB = contractedGaussianB.contracted_GTO.size();

        for (std::size_t ii = 0; ii < nA; ii++) {
            for (std::size_t jj = 0; jj < nB; jj++) {
                auto productGaussianAB = computeGaussianProduct(
                    contractedGaussianA.contracted_GTO[ii],
                    contractedGaussianA.location_x,
                    contractedGaussianA.location_y,
                    contractedGaussianA.location_z,
                    contractedGaussianB.contracted_GTO[jj],
                    contractedGaussianB.location_x,
                    contractedGaussianB.location_y,
                    contractedGaussianB.location_z
                );

                double value = computePrimitive3D(
                    contractedGaussianA.contracted_GTO[ii],
                    contractedGaussianA.location_x,
                    contractedGaussianA.location_y,
                    contractedGaussianA.location_z,
                    contractedGaussianA.shell_x,
                    contractedGaussianA.shell_y,
                    contractedGaussianA.shell_z,
                    contractedGaussianB.contracted_GTO[jj],
                    contractedGaussianB.location_x,
                    contractedGaussianB.location_y,
                    contractedGaussianB.location_z,
                    contractedGaussianB.shell_x,
                    contractedGaussianB.shell_y,
                    contractedGaussianB.shell_z,
                    productGaussianAB.gaussian_center,
                    productGaussianAB.gaussian_integral[3]
                );

                value *= contractedGaussianA.contracted_GTO[ii].orbital_coeff * contractedGaussianA.contracted_GTO[ii].orbital_norm;
                value *= contractedGaussianB.contracted_GTO[jj].orbital_coeff * contractedGaussianB.contracted_GTO[jj].orbital_norm;
                integral += value;
            }
        }

        return integral;
    }
};

struct Kinetic {
    static double computePrimitive3D(
        const Hermite::Primitive& primitiveA,
        double xA,
        double yA,
        double zA,
        std::int64_t lxA,
        std::int64_t lyA,
        std::int64_t lzA,
        const Hermite::Primitive& primitiveB,
        double xB,
        double yB,
        double zB,
        std::int64_t lxB,
        std::int64_t lyB,
        std::int64_t lzB,
        const std::array<double, 3>& gaussianCenter,
        double gaussianIntegral
    ) {
        // Compute overlap integrals for different angular momentum combinations
        const double overlapX = Overlap::computePrimitive1D(
            primitiveA.primitive_exp, xA, lxA,
            primitiveB.primitive_exp, xB, lxB,
            gaussianCenter[0]
        );
        const double overlapY = Overlap::computePrimitive1D(
            primitiveA.primitive_exp, yA, lyA,
            primitiveB.primitive_exp, yB, lyB,
            gaussianCenter[1]
        );
        const double overlapZ = Overlap::computePrimitive1D(
            primitiveA.primitive_exp, zA, lzA,
            primitiveB.primitive_exp, zB, lzB,
            gaussianCenter[2]
        );

        // Compute (-,+) combinations
        const double overlapXMP = Overlap::computePrimitive1D(
            primitiveA.primitive_exp, xA, (lxA > 0) ? lxA - 1 : 0,
            primitiveB.primitive_exp, xB, lxB + 1,
            gaussianCenter[0]
        );
        const double overlapYMP = Overlap::computePrimitive1D(
            primitiveA.primitive_exp, yA, (lyA > 0) ? lyA - 1 : 0,
            primitiveB.primitive_exp, yB, lyB + 1,
            gaussianCenter[1]
        );
        const double overlapZMP = Overlap::computePrimitive1D(
            primitiveA.primitive_exp, zA, (lzA > 0) ? lzA - 1 : 0,
            primitiveB.primitive_exp, zB, lzB + 1,
            gaussianCenter[2]
        );

        // Compute (+,+) combinations
        const double overlapXPP = Overlap::computePrimitive1D(
            primitiveA.primitive_exp, xA, lxA + 1,
            primitiveB.primitive_exp, xB, lxB + 1,
            gaussianCenter[0]
        );
        const double overlapYPP = Overlap::computePrimitive1D(
            primitiveA.primitive_exp, yA, lyA + 1,
            primitiveB.primitive_exp, yB, lyB + 1,
            gaussianCenter[1]
        );
        const double overlapZPP = Overlap::computePrimitive1D(
            primitiveA.primitive_exp, zA, lzA + 1,
            primitiveB.primitive_exp, zB, lzB + 1,
            gaussianCenter[2]
        );

        // Compute (-,-) combinations
        const double overlapXMM = Overlap::computePrimitive1D(
            primitiveA.primitive_exp, xA, (lxA > 0) ? lxA - 1 : 0,
            primitiveB.primitive_exp, xB, (lxB > 0) ? lxB - 1 : 0,
            gaussianCenter[0]
        );
        const double overlapYMM = Overlap::computePrimitive1D(
            primitiveA.primitive_exp, yA, (lyA > 0) ? lyA - 1 : 0,
            primitiveB.primitive_exp, yB, (lyB > 0) ? lyB - 1 : 0,
            gaussianCenter[1]
        );
        const double overlapZMM = Overlap::computePrimitive1D(
            primitiveA.primitive_exp, zA, (lzA > 0) ? lzA - 1 : 0,
            primitiveB.primitive_exp, zB, (lzB > 0) ? lzB - 1 : 0,
            gaussianCenter[2]
        );

        // Compute (+,-) combinations
        const double overlapXPM = Overlap::computePrimitive1D(
            primitiveA.primitive_exp, xA, lxA + 1,
            primitiveB.primitive_exp, xB, (lxB > 0) ? lxB - 1 : 0,
            gaussianCenter[0]
        );
        const double overlapYPM = Overlap::computePrimitive1D(
            primitiveA.primitive_exp, yA, lyA + 1,
            primitiveB.primitive_exp, yB, (lyB > 0) ? lyB - 1 : 0,
            gaussianCenter[1]
        );
        const double overlapZPM = Overlap::computePrimitive1D(
            primitiveA.primitive_exp, zA, lzA + 1,
            primitiveB.primitive_exp, zB, (lzB > 0) ? lzB - 1 : 0,
            gaussianCenter[2]
        );

        // Build kinetic energy components
        double kx = static_cast<double>(lxA * lxB) * overlapXMM;
        double ky = static_cast<double>(lyA * lyB) * overlapYMM;
        double kz = static_cast<double>(lzA * lzB) * overlapZMM;

        kx -= (2.0 * primitiveA.primitive_exp * static_cast<double>(lxB)) * overlapXPM;
        ky -= (2.0 * primitiveA.primitive_exp * static_cast<double>(lyB)) * overlapYPM;
        kz -= (2.0 * primitiveA.primitive_exp * static_cast<double>(lzB)) * overlapZPM;

        kx -= (2.0 * primitiveB.primitive_exp * static_cast<double>(lxA)) * overlapXMP;
        ky -= (2.0 * primitiveB.primitive_exp * static_cast<double>(lyA)) * overlapYMP;
        kz -= (2.0 * primitiveB.primitive_exp * static_cast<double>(lzA)) * overlapZMP;

        kx += (4.0 * primitiveB.primitive_exp * primitiveA.primitive_exp) * overlapXPP;
        ky += (4.0 * primitiveB.primitive_exp * primitiveA.primitive_exp) * overlapYPP;
        kz += (4.0 * primitiveB.primitive_exp * primitiveA.primitive_exp) * overlapZPP;

        constexpr double pi = 3.141592653589793238462643383279502884;
        const double combExp = primitiveA.primitive_exp + primitiveB.primitive_exp;
        const double prefactor = std::pow(pi / combExp, 1.5);

        const double Kx = 0.5 * kx * overlapY * overlapZ * prefactor;
        const double Ky = 0.5 * ky * overlapZ * overlapX * prefactor;
        const double Kz = 0.5 * kz * overlapX * overlapY * prefactor;

        return (Kx + Ky + Kz) * gaussianIntegral;
    }

    static double computeContracted(
        const Hermite::Contracted& contractedGaussianA,
        const Hermite::Contracted& contractedGaussianB
    ) {
        double integral = 0.0;

        const std::size_t nA = contractedGaussianA.contracted_GTO.size();
        const std::size_t nB = contractedGaussianB.contracted_GTO.size();

        for (std::size_t ii = 0; ii < nA; ii++) {
            for (std::size_t jj = 0; jj < nB; jj++) {
                auto productGaussianAB = computeGaussianProduct(
                    contractedGaussianA.contracted_GTO[ii],
                    contractedGaussianA.location_x,
                    contractedGaussianA.location_y,
                    contractedGaussianA.location_z,
                    contractedGaussianB.contracted_GTO[jj],
                    contractedGaussianB.location_x,
                    contractedGaussianB.location_y,
                    contractedGaussianB.location_z
                );

                double value = computePrimitive3D(
                    contractedGaussianA.contracted_GTO[ii],
                    contractedGaussianA.location_x,
                    contractedGaussianA.location_y,
                    contractedGaussianA.location_z,
                    contractedGaussianA.shell_x,
                    contractedGaussianA.shell_y,
                    contractedGaussianA.shell_z,
                    contractedGaussianB.contracted_GTO[jj],
                    contractedGaussianB.location_x,
                    contractedGaussianB.location_y,
                    contractedGaussianB.location_z,
                    contractedGaussianB.shell_x,
                    contractedGaussianB.shell_y,
                    contractedGaussianB.shell_z,
                    productGaussianAB.gaussian_center,
                    productGaussianAB.gaussian_integral[3]
                );

                value *= contractedGaussianA.contracted_GTO[ii].orbital_coeff * contractedGaussianA.contracted_GTO[ii].orbital_norm;
                value *= contractedGaussianB.contracted_GTO[jj].orbital_coeff * contractedGaussianB.contracted_GTO[jj].orbital_norm;
                integral += value;
            }
        }

        return integral;
    }
};

struct Nuclear {
    static double computePrimitive1D(
        std::int64_t indexA,
        std::int64_t indexB,
        double indexC,
        std::int64_t shellA,
        double centerA,
        std::int64_t shellB,
        double centerB,
        double atomCenter,
        double gaussianCenter,
        double gamma
    ) {
        // Placeholder implementation - needs to be completed based on Huzinaga nuclear integral formula
        // This is a stub for the incomplete function in the template
        return 0.0;
    }
};

} // namespace Huzinaga

PYBIND11_MODULE(planck_integrals, m) {
    m.doc() = "Planck integrals (Hermite and Huzinaga methods) via pybind11";

    py::class_<Hermite::Primitive>(m, "Primitive")
        .def(py::init<double, double, double>(),
             py::arg("exponent"),
             py::arg("coeff") = 1.0,
             py::arg("norm") = 1.0)
        .def_readwrite("primitive_exp", &Hermite::Primitive::primitive_exp)
        .def_readwrite("orbital_coeff", &Hermite::Primitive::orbital_coeff)
        .def_readwrite("orbital_norm", &Hermite::Primitive::orbital_norm);

    py::class_<Hermite::Contracted>(m, "Contracted")
        .def(py::init<>())
        .def_readwrite("location_x", &Hermite::Contracted::location_x)
        .def_readwrite("location_y", &Hermite::Contracted::location_y)
        .def_readwrite("location_z", &Hermite::Contracted::location_z)
        .def_readwrite("shell_x", &Hermite::Contracted::shell_x)
        .def_readwrite("shell_y", &Hermite::Contracted::shell_y)
        .def_readwrite("shell_z", &Hermite::Contracted::shell_z)
        .def_readwrite("contracted_GTO", &Hermite::Contracted::contracted_GTO);

    m.def(
        "overlap_primitive_1d",
        &Hermite::Overlap::computePrimitive1D,
        py::arg("exponentA"),
        py::arg("centerA"),
        py::arg("shellA"),
        py::arg("exponentB"),
        py::arg("centerB"),
        py::arg("shellB"),
        py::arg("hermiteNodes"),
        "Compute 1D Hermite overlap integral between two primitives."
    );

    m.def(
        "overlap_primitive_3d",
        &Hermite::Overlap::computePrimitive3D,
        py::arg("primitiveA"),
        py::arg("xA"),
        py::arg("yA"),
        py::arg("zA"),
        py::arg("lxA"),
        py::arg("lyA"),
        py::arg("lzA"),
        py::arg("primitiveB"),
        py::arg("xB"),
        py::arg("yB"),
        py::arg("zB"),
        py::arg("lxB"),
        py::arg("lyB"),
        py::arg("lzB"),
        "Compute 3D Hermite overlap integral between two primitives."
    );

    m.def(
        "overlap_contracted",
        &Hermite::Overlap::computeContracted,
        py::arg("contractedA"),
        py::arg("contractedB"),
        "Compute contracted overlap integral between two contracted Gaussians."
    );

    m.def(
        "kinetic_primitive_3d",
        &Hermite::Kinetic::computePrimitive3D,
        py::arg("primitiveA"),
        py::arg("xA"),
        py::arg("yA"),
        py::arg("zA"),
        py::arg("lxA"),
        py::arg("lyA"),
        py::arg("lzA"),
        py::arg("primitiveB"),
        py::arg("xB"),
        py::arg("yB"),
        py::arg("zB"),
        py::arg("lxB"),
        py::arg("lyB"),
        py::arg("lzB"),
        "Compute 3D Hermite kinetic integral between two primitives."
    );

    m.def(
        "kinetic_contracted",
        &Hermite::Kinetic::computeContracted,
        py::arg("contractedA"),
        py::arg("contractedB"),
        "Compute contracted kinetic integral between two contracted Gaussians."
    );

    // Huzinaga integrals
    py::class_<Huzinaga::HuzinagaGaussianProduct>(m, "HuzinagaGaussianProduct")
        .def(py::init<>())
        .def_readwrite("gaussian_center", &Huzinaga::HuzinagaGaussianProduct::gaussian_center)
        .def_readwrite("gaussian_integral", &Huzinaga::HuzinagaGaussianProduct::gaussian_integral)
        .def_readwrite("combined_exp", &Huzinaga::HuzinagaGaussianProduct::combined_exp)
        .def("__repr__", [](const Huzinaga::HuzinagaGaussianProduct& gp) {
            return "HuzinagaGaussianProduct(center=[" +
                   std::to_string(gp.gaussian_center[0]) + ", " +
                   std::to_string(gp.gaussian_center[1]) + ", " +
                   std::to_string(gp.gaussian_center[2]) + "], exp=" +
                   std::to_string(gp.combined_exp) + ")";
        });

    m.def(
        "huzinaga_overlap_primitive_1d",
        &Huzinaga::Overlap::computePrimitive1D,
        py::arg("exponentA"),
        py::arg("centerA"),
        py::arg("shellA"),
        py::arg("exponentB"),
        py::arg("centerB"),
        py::arg("shellB"),
        py::arg("gaussianCenter"),
        "Compute 1D Huzinaga overlap integral between two primitives."
    );

    m.def(
        "huzinaga_overlap_primitive_3d",
        [](const Hermite::Primitive& primitiveA,
           double xA, double yA, double zA,
           std::int64_t lxA, std::int64_t lyA, std::int64_t lzA,
           const Hermite::Primitive& primitiveB,
           double xB, double yB, double zB,
           std::int64_t lxB, std::int64_t lyB, std::int64_t lzB,
           const std::vector<double>& gaussianCenter,
           double gaussianIntegral) {
            if (gaussianCenter.size() != 3) {
                throw std::invalid_argument("gaussianCenter must have 3 elements");
            }
            std::array<double, 3> center = {gaussianCenter[0], gaussianCenter[1], gaussianCenter[2]};
            return Huzinaga::Overlap::computePrimitive3D(
                primitiveA, xA, yA, zA, lxA, lyA, lzA,
                primitiveB, xB, yB, zB, lxB, lyB, lzB,
                center, gaussianIntegral
            );
        },
        py::arg("primitiveA"),
        py::arg("xA"),
        py::arg("yA"),
        py::arg("zA"),
        py::arg("lxA"),
        py::arg("lyA"),
        py::arg("lzA"),
        py::arg("primitiveB"),
        py::arg("xB"),
        py::arg("yB"),
        py::arg("zB"),
        py::arg("lxB"),
        py::arg("lyB"),
        py::arg("lzB"),
        py::arg("gaussianCenter"),
        py::arg("gaussianIntegral"),
        "Compute 3D Huzinaga overlap integral between two primitives."
    );

    m.def(
        "huzinaga_overlap_contracted",
        &Huzinaga::Overlap::computeContracted,
        py::arg("contractedA"),
        py::arg("contractedB"),
        "Compute contracted Huzinaga overlap integral between two contracted Gaussians."
    );

    m.def(
        "huzinaga_kinetic_primitive_3d",
        [](const Hermite::Primitive& primitiveA,
           double xA, double yA, double zA,
           std::int64_t lxA, std::int64_t lyA, std::int64_t lzA,
           const Hermite::Primitive& primitiveB,
           double xB, double yB, double zB,
           std::int64_t lxB, std::int64_t lyB, std::int64_t lzB,
           const std::vector<double>& gaussianCenter,
           double gaussianIntegral) {
            if (gaussianCenter.size() != 3) {
                throw std::invalid_argument("gaussianCenter must have 3 elements");
            }
            std::array<double, 3> center = {gaussianCenter[0], gaussianCenter[1], gaussianCenter[2]};
            return Huzinaga::Kinetic::computePrimitive3D(
                primitiveA, xA, yA, zA, lxA, lyA, lzA,
                primitiveB, xB, yB, zB, lxB, lyB, lzB,
                center, gaussianIntegral
            );
        },
        py::arg("primitiveA"),
        py::arg("xA"),
        py::arg("yA"),
        py::arg("zA"),
        py::arg("lxA"),
        py::arg("lyA"),
        py::arg("lzA"),
        py::arg("primitiveB"),
        py::arg("xB"),
        py::arg("yB"),
        py::arg("zB"),
        py::arg("lxB"),
        py::arg("lyB"),
        py::arg("lzB"),
        py::arg("gaussianCenter"),
        py::arg("gaussianIntegral"),
        "Compute 3D Huzinaga kinetic integral between two primitives."
    );

    m.def(
        "huzinaga_kinetic_contracted",
        &Huzinaga::Kinetic::computeContracted,
        py::arg("contractedA"),
        py::arg("contractedB"),
        "Compute contracted Huzinaga kinetic integral between two contracted Gaussians."
    );
}

