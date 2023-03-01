from lsdo_rotor.vnv_scripts.axial_inflow.UIUC_validation import return_validation_model
from csdl_om import Simulator
import pytest


def test_UIUC_prop_validation():
    csdl_model = return_validation_model()
    sim = Simulator(csdl_model)
    sim.run()
    T = sim['T'].flatten()
    T_0 = T[0]
    T_end = T[-1]
    assert pytest.approx(T_end,rel=1e-3) == 0.54827259
    assert pytest.approx(T_0, rel=1e-3) == 10.42765851 # 10.427658510860592
    
    print(T_0)
    print(T_end)








#  [10.42765851 10.46168784 10.48909092 10.50872459 10.51687433 10.51537339
#  10.50423862 10.48467253 10.45515646 10.41320235 10.360162   10.29750014
#  10.22396721 10.13845843 10.04284755  9.93707095  9.82027404  9.69331275
#   9.5564758   9.40864574  9.24678643  9.06703858  8.86717956  8.64725686
#   8.40892141  8.15463876  7.88700941  7.60835029  7.32092992  7.02652209
#   6.7267679   6.42282949  6.11535265  5.80493824  5.49194665  5.18161169
#   4.86743415  4.54955636  4.22761137  3.90291212  3.57598865  3.2476252
#   2.91860358  2.58706187  2.25229516  1.91497017  1.57574304  1.23507796
#   0.893004    0.54827259]