from __future__ import annotations

from copy import deepcopy
from math import inf
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from rich.live import Live
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import concreteproperties.results as res
import concreteproperties.stress_strain_profile as ssp
import concreteproperties.utils as utils
from concreteproperties.design_codes.design_code import DesignCode
from concreteproperties.material import Concrete, SteelBar

if TYPE_CHECKING:
    from concreteproperties.concrete_section import ConcreteSection


class ACI318M(DesignCode):
    """Design code class for American standard ACI 318-19 in metric units.

    .. note::
        Note that this design code only supports
        :class:`~concreteproperties.material.Concrete` and
        :class:`~concreteproperties.material.SteelBar` material objects. Meshed
        :class:`~concreteproperties.material.Steel` material objects and
        :class:'~concreteproperties.material.SteelStrand' material objects are 
        **not** currently supported.
    """

    def __init__(self):
        """Inits the ACI318M class."""
        self.analysis_code = "ACI 318M-19"
        super().__init__()

    def assign_concrete_section(
        self,
        concrete_section: ConcreteSection, tie_type:str = "ties"
    ):
        """Assigns a concrete section to the design code.

        :param concrete_section: Concrete section object to analyse
        """

        self.concrete_section = concrete_section

        # check to make sure there are no meshed reinforcement regions
        if self.concrete_section.reinf_geometries_meshed:
            raise ValueError(
                "Meshed reinforcement is not supported in this design code."
            )

        #calculate squash and tensile load
        squash, tensile = self.squash_tensile_load(tie_type)
        self.squash_load = squash
        self.tensile_load = tensile
        
    def e_conc(self, compressive_strength: float, density: float = 2400) -> float:
        r"""Calculates Youngs Modulus (:math:`E_c`) for concrete in accordance with
        ACI 318-19 19.2.2.1a.
        :math:`E_c=\displaystyle\rho^{1.5}*0.041\sqrt{f'c}`
        :param compressive_strength: 28 day compressive concrete strength (MPa)
        :param density: Concrete density :math:`\rho`, defaults to 2400 kg/m\ :sup:`3`
        for normal weight concrete
        :return: :math:`E_c`, Youngs Modulus (MPa)
        """
        # Check low and high limit on density in ACI 318-19 19.2.2.1a for E_c equation
        # to be valid
        low_limit = 1440
        high_limit = 2565

        # check upper and lower concrete strengths
        self.check_density_limits(density, low_limit, high_limit)

        E_c = (density**1.5) * 0.041 * np.sqrt(compressive_strength)

        return E_c
   
    def check_f_c_limits(
        self, compressive_strength: float, low_limit: float=17.25, high_limit: float=69.0
    ) -> None:
        r"""Checks that the compressive strength is within the supported limits.
        :param compressive_strength: 28 day compressive concrete strength (MPa)
        :param low_limit: Lower limit for compressive strength from ACI 318-19 19.2.1.1
        :param high_limit: Upper limit for compressive strength per program limits
        :raises ValueError: If compressive strength is outside of supported limits
        """
        if compressive_strength < low_limit:
            raise ValueError(f"compressive_strength must be at least {low_limit}MPa per 19.2.1.1")
        #Refer to ACI 318-19 19.2.1.1 for minimum concrete compressive strengths per element.
        #The general limitation of 2500psi or 17.25MPa holds for most non-seismic construction.
        #Some structures and elements will require a highter minimum compressive strength.
        
        if compressive_strength > high_limit:
            raise ValueError(f"compressive_strength greater than {high_limit}MPa is not supported")
        #Concrete compressive strengths above 69 MPa will have limitations 
        #on shear strength per ACI 318-19 22.5.3.1, 22.7.2.1, etc. and additional requirements
        #for earthquake resistaning special moment frames.


    def check_density_limits(
        self, density: float, low_limit: float, high_limit: float
    ) -> None:
        r"""Checks that the density is within the bounds outlined
        for the elastic modulus expression ACI 318-19 19.2.2.1a to be valid.
        :param density: Concrete density :math:`\rho` 
        :param low_limit: Lower limit for density from ACI 318-19 19.2.2.1a
        :param high_limit: Upper limit for density from ACI 318-19 19.2.2.1a
        :raises ValueError: If density is outside of the limits within ACI 318-19 19.2.2.1a
        """
        if not (low_limit <= density <= high_limit):
            raise ValueError(
                f"The specified concrete density of {density}kg/m^3 is not within the "
                f"bounds of {low_limit}kg/m^3 & {high_limit}kg/m^3 for the "
                f"{self.analysis_code} Elastic Modulus eqn to be applicable"
            )

    def alpha_1(self) -> float:
        r"""Scaling factor relating the nominal 28 day concrete compressive strength to
        the effective concrete compressive strength used for design purposes within the
        concrete stress block. For rectangular "Whitney" stress block in accordance with 
        ACI 318-19 22.2.2.4.1, use 0.85.
        :param compressive_strength: 28 day compressive design strength (MPa)
        :return: :math:`\alpha_1` factor
        """

        alpha_1 = 0.85

        return alpha_1

    def beta_1(self, compressive_strength: float) -> float:
        r"""Scaling factor relating the depth of an equivalent rectangular compressive
        stress block (:math:`a`) to the depth of the neutral axis (:math:`c`).
        A function of the concrete compressive strength, per ACI 318-19 22.2.2.4.3.
        :math:`\displaystyle\quad\beta_1=0.85-\frac{0.05(f'_c-28)}{7}\quad:0.65\leq\beta_1\leq0.85`
        :param compressive_strength: 28 day compressive design strength (MPa)
        :return: :math:`\beta_1` factor
        """

        beta1 = 0.85 - 0.05 * (compressive_strength-28) / 7
        beta1 = min(beta1, 0.85)
        beta1 = max(beta1, 0.65)
        # corresponds to gamma in AS3600 and base code

        return beta_1
   

    def lamda(self, density: float) -> float:
        r"""Modification factor reflecting the reduced mechanical properties of
        lightweight concrete relative to normal weight concrete of the same compression
        strength, in accordance with ACI 318-19 19.2.4.1
        :math:`\displaystyle\quad\lambda=\frac{\rho}{2133}\quad:0.75\leq\lambda\leq1.0`
        :param density: Equilibrium density of concrete mixture
        :return: :math:`\lambda` factor
        """
        
        lamda = density / 2133
        lamda = min(lamda, 1.0)
        lamda = max(lamda, 0.75)

        return lamda

    def modulus_of_rupture(
        self,
        compressive_strength: float,
        density: float = 2400,
    ) -> float:
        r"""Calculates the average modulus of rupture of concrete (:math:`f_r`) in
        accordance with ACI 318-19 19.2.3.1 for deflection calculations.
        :math:`\quad f_r=0.62\lambda({f'_c})^{0.5}`
        :param compressive_strength: 28 day compressive design strength (MPa)
        :param density: Density of concrete material
        :return: Modulus of rupture (:math:`f_r`)
        """

        f_r = 0.62 * self.lamda(density) * np.sqrt(compressive_strength)

        return f_r

    def create_concrete_material(
        self,
        compressive_strength: float,
        density: float = 2400,
        colour: str = "lightgrey",
    ) -> Concrete:
        r"""Returns a concrete material object to ACI 318M-19.

        .. admonition:: Material assumptions

          - *Density*: entered in kg/m\ :sup:`3`, default 2400 kg/m\ :sup:`3`
          
          - *Elastic modulus*: Calculated from ACI 318-19 19.2.2.1a
          
          - *Service stress-strain profile*: Linear with no tension, compressive
            strength at :math:`0.85f'_c`

          - *Ultimate stress-strain profile*: Rectangular stress block per ACI 318-19 22.2.2.4.1

        :param compressive_strength: Characteristic compressive strength of
            concrete at 28 days in megapascals (MPa)
            
        :param density: Density of concrete in kg/m\ :sup:`3`
            
        :param colour: Colour of the concrete for rendering

        :return: Concrete material object
        """

        self.check_density_limits(density)
        
        # create concrete name
        name = f"{compressive_strength:.0f} MPa Concrete \n({self.analysis_code})"

         # calculate elastic modulus
        elastic_modulus = self.e_conc(compressive_strength, density)

        # calculate rectangular stress block parameters
        alpha_1 = self.alpha_1()
        beta_1 = self.beta_1(compressive_strength)
        
        # calculate lightweight concrete factor lambda
        lamda = self.lamda(density)

        return Concrete(
            name=name,
            density=density*1e-9,
            stress_strain_profile=ssp.ConcreteLinearNoTension(
                elastic_modulus=elastic_modulus,
                ultimate_strain=0.003,
                compressive_strength=alpha_1 * compressive_strength,
            ),
            ultimate_stress_strain_profile=ssp.RectangularStressBlock(
                compressive_strength=compressive_strength,
                alpha=alpha,
                gamma=beta1,
                ultimate_strain=0.003,
            ),
            flexural_tensile_strength=0,
            colour=colour,
        )

    def create_steel_material(
        self,
        yield_strength: float = 413,
        colour: str = "grey",
    ) -> SteelBar:
        r"""Returns a steel bar material object.

        .. admonition:: Material assumptions

          - *Density*: 7850 kg/m\ :sup:`3`

          - *Elastic modulus*: 200000 MPa

          - *Stress-strain profile*: Elastic-plastic, fracture strain from ASTM A615

        :param yield_strength: Steel yield strength
        :param colour: Colour of the steel for rendering

        :return: Steel material object
        """

        if yield_strength <= 425:
            fracture_strain = 0.07
        else:
            fracture_strain = 0.06
        # maximum elongation for ASTM A615 Gr. 60 bar sizes 9 to 18
        # Lower grade and smaller bar sizes may have larger elongations

        return SteelBar(
            name=f"{yield_strength:.0f} MPa Steel \n({self.analysis_code})",
            density=7.85e-6,
            stress_strain_profile=ssp.SteelElasticPlastic(
                yield_strength=yield_strength,
                elastic_modulus=200e3,
                fracture_strain=fracture_strain,
            ),
            colour=colour,
        )

    def squash_tensile_load(self, tie_type:str = "ties") -> Tuple[float, float]:
        """Calculates the squash and tensile load of the reinforced concrete section.

        :return: Squash and tensile load
        """

        # initialise the P_0, squash load, and tensile load variables
        P_0 = 0
        squash_load = 0
        tensile_load = 0

        # loop through all concrete geometries
        for conc_geom in self.concrete_section.concrete_geometries:
            # calculate area
            area = conc_geom.calculate_area()

            # calculate compressive force
            force_c = (
                area
                * conc_geom.material.ultimate_stress_strain_profile.get_compressive_strength()
            )

            # add to totals
            P_0 += force_c

        # loop through all steel geometries
        for steel_geom in self.concrete_section.reinf_geometries_lumped:
            # calculate area
            area = steel_geom.calculate_area()
            # calculate capacity of steel bars in compression per ACI 318-19 22.4.2.1
            yield_c = steel_geom.material.stress_strain_profile.get_yield_strength()
            yield_c = max(yield_c, 550)

            # calculate compressive and tensile force
            force_c = area * yield_c
            
            force_t = -area * steel_geom.material.stress_strain_profile.get_yield_strength()

            # add to totals
            P_0 += force_c
            tensile_load += force_t
            
        #limit maximum axial strength (P_n,max) per Table 22.4.2.1
        if tie_type == "spiral":
            squash_load = 0.85*P_0
        else:
            squash_load = 0.80*P_0

        return squash_load, tensile_load 

## Bookmark -- Lo is here     
    
    def capacity_reduction_factor(
        self,
        theta: float,
        tie_type: str ="ties",
    ) -> float:
        """Returns the ACI 318-19 strength reduction factor phi (Table 21.2.2).
        
        :param theta: Angle (in radians) the neutral axis makes with the
        horizontal axis (:math:`-\pi \leq \theta \leq \pi`)
        :param tie_type: Type of transverse reinforcement in the member. Designating
        "spiral" reinforcement will allow larger capacity factor. Defaults to "ties".

        :return: Capacity reduction factor
        """

        # phi for tension-controlled sections (e.g. pure bending)
        phi_t = 0.90
        
        # phi for compression-controlled sections
        if tie_type == "spirals":
            phi_c = 0.75
        else:
            phi_c = 0.65
            
        strain_ty = (
            extreme_geom.material.stress_strain_profile.get_yield_strength()/
            extreme_geom.material.stress_strain_profile.get_elastic_modulus()+0.003
        )
## check this!        
        strain_t = extreme_geom.material.stress_strain_profile.get_ultimate_tensile_strain
## check this!
            
        phi = phi_c + (phi_t-phi_c)*(strain_t-strain_ty)/0.003
        phi = min(phi, phi_t)
        phi = max(phi, phi_c)
        
        return phi
        
    def get_k_uo(
        self,
        theta: float,
    ) -> float:
        r"""Returns k_uo for the reinforced concrete cross-section given ``theta``.

        :param theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)

        :return: Bending parameter k_uo
        """

        pure_res = self.concrete_section.ultimate_bending_capacity(theta=theta)

        return pure_res.k_u

    def get_n_ub(
        self,
        theta: float,
    ) -> float:
        r"""Returns n_ub for the reinforced concrete cross-section given ``theta``.

        :param theta: Angle (in radians) the neutral axis makes with the
            horizontal axis (:math:`-\pi \leq \theta \leq \pi`)

        :return: Balanced axial force n_ub
        """

        # get depth to extreme tensile bar and its yield strain
        d_0, eps_sy = self.concrete_section.extreme_bar(theta=theta)

        # get compressive strain at extreme fibre
        eps_cu = self.concrete_section.gross_properties.conc_ultimate_strain

        # calculate d_n at balanced load
        d_nb = d_0 * (eps_cu) / (eps_sy + eps_cu)

        # calculate axial force at balanced load
        balanced_res = self.concrete_section.calculate_ultimate_section_actions(
            d_n=d_nb, ultimate_results=res.UltimateBendingResults(theta=theta)
        )

        return balanced_res.n

    def ultimate_bending_capacity(
        self,
        theta: float = 0,
        n_design: float = 0,
        phi_0: float = 0.6,
    ) -> Tuple[res.UltimateBendingResults, res.UltimateBendingResults, float]:
        r"""Calculates the ultimate bending capacity with capacity factors to
        AS 3600:2018.

        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        :param n_design: Design axial force, N*
        :param phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)

        :return: Factored and unfactored ultimate bending results objects, and capacity
            reduction factor *(factored_results, unfactored_results, phi)*
        """

        # get parameters to determine phi
        n_uot = self.tensile_load
        k_uo = self.get_k_uo(theta=theta)
        n_ub = self.get_n_ub(theta=theta)

        # non-linear calculation of phi
        def non_linear_phi(phi_guess):
            phi = self.capacity_reduction_factor(
                n_u=n_design / phi_guess,
                n_ub=n_ub,
                n_uot=n_uot,
                k_uo=k_uo,
                phi_0=phi_0,
            )

            return phi - phi_guess

        phi, _ = brentq(
            f=non_linear_phi,
            a=phi_0,
            b=0.85,
            xtol=1e-3,
            rtol=1e-6,  # type: ignore
            full_output=True,
            disp=False,
        )

        # generate basic moment interaction diagram
        f_mi_res, _, _ = self.moment_interaction_diagram(
            theta=theta,
            control_points=[
                ("N", 0.0),
            ],
            n_points=2,
            phi_0=phi_0,
            progress_bar=False,
        )

        # get significant axial loads
        n_squash = f_mi_res.results[0].n
        n_decomp = f_mi_res.results[1].n
        n_tensile = f_mi_res.results[-1].n

        # DETERMINE where we are on interaction diagram
        # if we are above the squash load or tensile load
        if n_design > n_squash:
            raise utils.AnalysisError(
                f"N = {n_design} is greater than the squash load, phiNc = {n_squash}."
            )
        elif n_design < n_tensile:
            raise utils.AnalysisError(
                f"N = {n_design} is greater than the tensile load, phiNt = {n_tensile}"
            )
        # compression linear interpolation
        elif n_design > n_decomp:
            factor = (n_design - n_decomp) / (n_squash - n_decomp)
            squash = f_mi_res.results[0]
            decomp = f_mi_res.results[1]
            ult_res = res.UltimateBendingResults(
                theta=theta,
                d_n=inf,
                k_u=0,
                n=n_design / phi,
                m_x=(decomp.m_x + factor * (squash.m_x - decomp.m_x)) / phi,
                m_y=(decomp.m_y + factor * (squash.m_y - decomp.m_y)) / phi,
                m_xy=(decomp.m_xy + factor * (squash.m_xy - decomp.m_xy)) / phi,
            )
        # regular calculation
        elif n_design > 0:
            ult_res = self.concrete_section.ultimate_bending_capacity(
                theta=theta, n=n_design / phi
            )
        # tensile linear interpolation
        else:
            factor = n_design / n_tensile
            pure = f_mi_res.results[-2]
            ult_res = res.UltimateBendingResults(
                theta=theta,
                d_n=inf,
                k_u=0,
                n=n_design / phi,
                m_x=(1 - factor) * pure.m_x / phi,
                m_y=(1 - factor) * pure.m_y / phi,
                m_xy=(1 - factor) * pure.m_xy / phi,
            )

        # factor ultimate results
        f_ult_res = deepcopy(ult_res)
        f_ult_res.n *= phi
        f_ult_res.m_x *= phi
        f_ult_res.m_y *= phi
        f_ult_res.m_xy *= phi

        return f_ult_res, ult_res, phi

    def moment_interaction_diagram(
        self,
        theta: float = 0,
        limits: List[Tuple[str, float]] = [
            ("D", 1.0),
            ("N", 0.0),
        ],
        control_points: List[Tuple[str, float]] = [
            ("fy", 1.0),
        ],
        labels: Optional[List[str]] = None,
        n_points: int = 24,
        n_spacing: Optional[int] = None,
        phi_0: float = 0.6,
        progress_bar: bool = True,
    ) -> Tuple[res.MomentInteractionResults, res.MomentInteractionResults, List[float]]:
        r"""Generates a moment interaction diagram with capacity factors to
        AS 3600:2018.

        See :meth:`concreteproperties.concrete_section.ConcreteSection.moment_interaction_diagram`
        for allowable control points.

        .. note::

            When providing ``"N"`` to ``limits`` or ``control_points``, ``"N"`` is taken
            to be the unfactored net (nominal) axial load :math:`N^{*} / \phi`.

        :param theta: Angle (in radians) the neutral axis makes with the horizontal axis
            (:math:`-\pi \leq \theta \leq \pi`)
        :param limits: List of control points that define the start and end of the
            interaction diagram. List length must equal two. The default limits range
            from concrete decompression strain to the pure bending point.
        :param control_points: List of additional control points to add to the moment
            interaction diagram. The default control points include the balanced point
            (``fy=1``). Control points may lie outside the limits of the moment
            interaction diagram as long as equilibrium can be found.
        :param labels: List of labels to apply to the ``limits`` and ``control_points``
            for plotting purposes. The first two values in ``labels`` apply labels to
            the ``limits``, the remaining values apply labels to the ``control_points``.
            If a single value is provided, this value will be applied to both ``limits``
            and all ``control_points``. The length of ``labels`` must equal ``1`` or
            ``2 + len(control_points)``.
        :param n_points: Number of points to compute including and between the
            ``limits`` of the moment interaction diagram. Generates equally spaced
            neutral axes between the ``limits``.
        :param n_spacing: If provided, overrides ``n_points`` and generates the moment
            interaction diagram using ``n_spacing`` equally spaced axial loads. Note
            that using ``n_spacing`` negatively affects performance, as the neutral axis
            depth must first be located for each point on the moment interaction
            diagram.
        :param phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)
        :param progress_bar: If set to True, displays the progress bar

        :return: Factored and unfactored moment interaction results objects, and list of
            capacity reduction factors *(factored_results, unfactored_results, phis)*
        """

        mi_res = self.concrete_section.moment_interaction_diagram(
            theta=theta,
            limits=limits,
            control_points=control_points,
            labels=labels,
            n_points=n_points,
            n_spacing=n_spacing,
            progress_bar=progress_bar,
        )

        # get theta
        theta = mi_res.results[0].theta

        # add squash load
        mi_res.results.insert(
            0,
            res.UltimateBendingResults(
                theta=theta,
                d_n=inf,
                k_u=0,
                n=self.squash_load,
                m_x=0,
                m_y=0,
                m_xy=0,
            ),
        )

        # add tensile load
        mi_res.results.append(
            res.UltimateBendingResults(
                theta=theta,
                d_n=0,
                k_u=0,
                n=self.tensile_load,
                m_x=0,
                m_y=0,
                m_xy=0,
            )
        )

        # make a copy of the results to factor
        f_mi_res = deepcopy(mi_res)

        # list to store phis
        phis = []

        # get required constants for phi
        n_uot = self.tensile_load
        k_uo = self.get_k_uo(theta=theta)
        n_ub = self.get_n_ub(theta=theta)

        # factor results
        for ult_res in f_mi_res.results:
            phi = self.capacity_reduction_factor(
                n_u=ult_res.n, n_ub=n_ub, n_uot=n_uot, k_uo=k_uo, phi_0=phi_0
            )
            ult_res.n *= phi
            ult_res.m_x *= phi
            ult_res.m_y *= phi
            ult_res.m_xy *= phi
            phis.append(phi)

        return f_mi_res, mi_res, phis

    def biaxial_bending_diagram(
        self,
        n_design: float = 0,
        n_points: int = 48,
        phi_0: float = 0.6,
        progress_bar: bool = True,
    ) -> Tuple[res.BiaxialBendingResults, List[float]]:
        """Generates a biaxial bending with capacity factors to AS 3600:2018.

        :param n_design: Design axial force, N*
        :param n_points: Number of calculation points
        :param phi_0: Compression dominant capacity reduction factor, see Table 2.2.2(d)
        :param progress_bar: If set to True, displays the progress bar

        :return: Factored biaxial bending results object and list of capacity reduction
            factors *(factored_results, phis)*
        """

        # initialise results
        f_bb_res = res.BiaxialBendingResults(n=n_design)
        phis = []

        # calculate d_theta
        d_theta = 2 * np.pi / n_points

        # generate list of thetas
        theta_list = np.linspace(start=-np.pi, stop=np.pi - d_theta, num=n_points)

        # function that performs biaxial bending analysis
        def bbcurve(progress=None):
            # loop through thetas
            for theta in theta_list:
                # factored capacity
                f_ult_res, _, phi = self.ultimate_bending_capacity(
                    theta=theta, n_design=n_design, phi_0=phi_0
                )
                f_bb_res.results.append(f_ult_res)
                phis.append(phi)

                if progress:
                    progress.update(task, advance=1)

        if progress_bar:
            # create progress bar
            progress = utils.create_known_progress()

            with Live(progress, refresh_per_second=10) as live:
                task = progress.add_task(
                    description="[red]Generating biaxial bending diagram",
                    total=n_points,
                )

                bbcurve(progress=progress)

                progress.update(
                    task,
                    description="[bold green]:white_check_mark: Biaxial bending diagram generated",
                )
                live.refresh()
        else:
            bbcurve()

        # add first result to end of list top
        f_bb_res.results.append(f_bb_res.results[0])
        phis.append(phis[0])

        return f_bb_res, phis
