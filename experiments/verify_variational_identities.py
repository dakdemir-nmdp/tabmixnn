"""Symbolic checks for the manuscript's finite-dimensional Gaussian appendix."""
from pathlib import Path
import json
import sympy as sp


def main():
    p,q,r,d1,d2=sp.symbols('p q r d1 d2',positive=True)
    # Domain: P symmetric positive definite, hence p*r > q**2.
    P=sp.Matrix([[p,q],[q,r]])
    D=sp.diag(d1,d2)
    m1,m2,h1,h2=sp.symbols('m1 m2 h1 h2',real=True)
    m=sp.Matrix([m1,m2]); h=sp.Matrix([h1,h2])
    objective=(m.T*P*m)[0]/2-(h.T*m)[0]+sp.trace(P*D)/2-sp.log(d1*d2)/2
    grad=sp.Matrix([sp.diff(objective,z) for z in [m1,m2]])
    assert sp.simplify(grad-(P*m-h))==sp.zeros(2,1)
    assert all(sp.simplify(sp.diff(objective,z).subs({d1:1/p,d2:1/r}))==0 for z in [d1,d2])
    P0=sp.Matrix([[2,1],[1,2]]); exact=P0.inv(); diagonal=sp.eye(2)/2
    plus=sp.ones(2,1); minus=sp.Matrix([1,-1])
    out=dict(domain='2x2 SPD posterior precision; positive diagonal variational variances',
        verdict='Symbolically verified gradients and exact counterexample; not a formal general theorem proof',
        mean_gradient=str(grad),optimal_mean='P^{-1} h',optimal_diagonal=['1/p','1/r'],
        exact_covariance=str(exact),variational_covariance=str(diagonal),
        kl_gap=str(sp.log(sp.Rational(4,3))/2),
        predictive_variances={name:dict(exact=str((v.T*exact*v)[0]),mean_field=str((v.T*diagonal*v)[0]))
                             for name,v in [('sum',plus),('difference',minus)]},
        sympy_version=sp.__version__)
    path=Path(__file__).resolve().parent/'results/variational_identities.json'
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
