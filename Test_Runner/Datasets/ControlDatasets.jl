
function InvertedPendulum()
    z1,z2 = symbols("z1,z2")
    xsym = [z1,z2]
    u = symbols("u")
    b, m, l, g = symbols("b,m,l,g")
    hyps = [b,m,l,g]
    hypvalues = [-0.2, 0.5, 0.6, 9.81]

    timestep = 0.075
    generate_distraction = () -> generate_distraction_from_gp(0.2 ;scale = 3.0)

    # We draw the control disturbations form a Gp.


    arguments= ( reltol = 1e-4, abstol = 1e-4, init_noise = 0.0, init = [0.0, pi], clip_ctrl = 20.0, R = ones(1,1) * 10.0, Q= Diagonal([1.0,10.0]), refstate = [0.0 , pi])


    I = 1/12 * m * l^2
    # We provide the decomposition F(x) *x!!!
    F = [0.0 -1/2 *m*l*g*sin(z2)//(z2*(1/4*m*l^2+I));
            1.0 0.0]
    FB = [1.0/(1/4*m*l^2+I);
        0.0]


    hp = ControlSystemDataSet( xsym, [u], F, FB, timestep, arguments; hyp = hyps, hyp_vals= hypvalues, generate_distraction = generate_distraction)
    return Gaussian( hp, "Inverted_Pendulum")
end


function CartPole()

    z1,z2,z3,z4 = symbols("z1,z2,z3,z4")
    xsym = [    z1,z2,z3,z4]
    u = symbols("u")
    b, m1,m2, l, g = symbols("b,m1,m2,l,g")
    hyps = [m1,m2,l,g]
    hypvalues = [0.5, 0.5, 0.6, 9.81]

    timestep = 0.1


    # We draw the control disturbations form a Gp.



    div = 4*(m1+m2) -3 *m2 *cos(z4)^2
    F = [0.0 1.0 0.0 0.0;
        0.0 0.0 (2 * m2 *l *z3 *sin(z4))/div  3*m2*g*sin(z4)*cos(z4)/(z4*div);
        0.0 0.0 -3*m2*l*z3*sin(z4)*cos(z4) -(6*(m1+m2)*g*sin(z4))/(z4*div);
        0.0 0.0 1.0 0.0]

    FB = [0.0; 4 /div; -6*cos(z4)/div; 0.0]


    generate_distraction = () -> generate_distraction_from_gp(0.2 ;scale = 3.0)
    # We draw the control disturbations form a Gp.


    arguments= ( reltol = 1e-4, abstol = 1e-4, init_noise = 0.0, init = [0.0, 0.0, 0.0, pi], clip_ctrl = 5.0, R = ones(1,1) * 10.0, Q= Diagonal([100.0,1.0, 1.0, 100.0]), refstate = [0.0, 0.0, 0.0, pi])

    hp = ControlSystemDataSet( xsym, [u], F, FB, timestep, arguments; hyp = hyps, hyp_vals= hypvalues, generate_distraction = generate_distraction)
    return Gaussian( hp, "CartPole")

end



function Pendubot()

    z1,z2,z3,z4 = symbols("z1,z2,z3,z4")
    xsym = [    z1,z2,z3,z4]
    u = symbols("u")
    m2,m3, l2,l3, g = symbols("m2,m3,l2,l3,g")
    hyps = [m2,m3,l2,l3,g]
    hypvalues = [0.5, 0.5, 0.6, 0.6, 9.81]

    timestep = 0.075


    # We draw the control disturbations form a Gp.



    F = [0.0 1.0 0.0 0.0;
        +g*l2*sin(z1)/z1*(0.5*m2+m3) 0.0 0.0 -l2*0.5*m3*l3*z4*sin(z1-z3);
        0.0 0.0 0.0 1.0;
        0.0 (0.5*m3*l3*(l2*z2*sin(z1-z3))) 0.5*m3*l3*g*sin(z3)/z3 0.0]

    FB = [0.0; 1.0; 0.0;  0.0]

    I2 = 1/12*m2*l2^2
    I3 = 1/12*m3*l3^2

    C =[ 1.0 0.0 0.0 0.0;
    0.0 l2^2*(0.25*m2+m3)+I2 0.0 0.5*m3*l3*l2*cos(z1-z3);
    0.0 0.0 1.0 0.0;
    0.0 0.5*l2*l3*m3*cos(z1-z3) 0.0 0.25*m3*l3^2+I3]



    generate_distraction = () -> generate_distraction_from_gp(;scale = 0.3)
    # We draw the control disturbations form a Gp.


    arguments= ( reltol = 1e-4, abstol = 1e-4, init_noise = 0.0, init = [pi, 0.0, pi, 0.0 ], clip_ctrl = 6.5, R = ones(1,1) * 10.0, Q= Diagonal([10.0,1.0, 100.0, 1.0]), refstate = [pi, 0.0,  pi, 0.0])

    hp = ControlSystemDataSet( xsym, [u], F, FB, timestep, arguments; hyp = hyps, hyp_vals= hypvalues, generate_distraction = generate_distraction, C = C)
    return Gaussian( hp, "Pendubot")

end







function CartDoublePendulum()

    z1,z2,z3,z4,z5,z6 = symbols("z1,z2,z3,z4,z5,z6")
    xsym = [z1,z2,z3,z4,z5,z6]
    u = symbols("u")
    m1, m2,m3, l2,l3, g = symbols("m1,m2,m3,l2,l3,g")

    hyps = [m1, m2,m3,l2,l3,g]
    hypvalues = [0.5, 0.5, 0.5, 0.6, 0.6, 9.81]

    timestep = 0.01


    # We draw the control disturbations form a Gp.


    FB = [0.0; 1.0; 0.0; 0.0; 0.0; 0.0]

    I2 = 1/12*m2*l2^2
    I3 = 1/12*m3*l3^2

    F = [0.0 1.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 -0.5*(m2+2*m3)*l2*z4*sin(z3) 0.0 -0.5*m3*l3*z6*sin(z5);
        0.0 0.0 0.0 1.0 0.0 0.0;
        0.0 0.0 (0.5*m2+m3)*l2*g*sin(z3)/z3 0.0 0.0 -0.5*m3*l2*l3*z6*sin(z3-z5);
        0.0 0.0 0.0 0.0 0.0 1.0;
        0.0 0.0 0.0 l2*z4*sin(z3-z5)*0.5*m3*l3 0.5*m3*l3*g*sin(z5)/z5 0.0]


    C = [1.0 0.0 0.0 0.0 0.0 0.0
        0.0 (m1+m2+m3) 0.0 -0.5*(m2+2m3)*l2*sin(z3) 0.0 -0.5*m3*l3*cos(z5);
        0.0 0.0 1.0 0.0 0.0 0.0;
        0.0 -(0.5*m2+m3)*l2*cos(z3) 0.0 (m3*l2^2 + I2 + 0.25*m2*l2^2) 0.0 0.5*m3*l2*l3*cos(z3-z5);
        0.0 0.0 0.0 0.0 1.0 0.0;
        0.0 -0.5*m3*l3*cos(z5) 0.0  0.5*m3*l2*l3*cos(z3-z5) 0.0 (0.25*m2*l3^2+I3)]


    #generate_distraction = ()  -> generate_distraction_from_gp(1.0, Stheno.Exp();scale = 4.0)
    generate_distraction = ()  -> generate_distraction_from_gp(0.2; scale = 3.0)

    # We draw the control disturbations form a Gp.


    arguments= (reltol = 1e-4, abstol = 1e-4,  init_noise = 0.0, init = [0.0, 0.0, pi, 0.0, pi, 0.0], clip_ctrl = 10, R = ones(1,1) * 10.0, Q= Diagonal([10.0, 1.0, 10.0, 1.0, 10.0, 1.0]), refstate = [0.0, 0.0,  pi, 0.0, pi, 0.0])

    hp = ControlSystemDataSet( xsym, [u], F, FB, timestep, arguments; hyp = hyps, hyp_vals= hypvalues, generate_distraction = generate_distraction, C= C, control = false)
    return Gaussian( hp, "CartDoublePendulum")

end





function Unicycle()

    u1,u2,u3,u4,u5,u6,u7 = symbols("u1,u2,u3,u4,u5,u6,u7")
    xsym = [u1,u2,u3,u4,u5,u6, u7]
    c1,c2 = symbols("c1,c2")
    a0dot, g, F = symbols("a0dot, g, F")

    hyps = [a0dot, g, F]
    hypvalues = [0.6, 9.81, 0.01]

    timestep = 0.05


    # We draw the control disturbations form a Gp.



    # Paper https://link.springer.com/article/10.1023/A:1026481216262

    A = [0 14.6*u2*u3 -194.6 0 0 (3.54*u3*u5*u1-2.7*u4 + 30.1*u3*u2*u5) 0;
           0 0 66 0 0 (0.75*u6*u3-1.77*u4-1.78*u3*u1*u5) 0;
           0 1 0 0 0 0 0;
           0 0 0 0 1.1*g (1.53*u2+0.22*(u1+a0dot)) 0;
           0 0 0 1 0 0 0 ;
           -0.1*u4 -1.4*u4 1.7*g*u5 -0.1*a0dot 0 (0.43*u3*(u1+a0dot) -2.6*F-0.81*u2*u3+0.53*u5*u4+0.37*u3*u7*u5) 0.37*u2*u3;
           0.1*u4 0.21*u4 -2.9*g*u5 0.1*a0dot 0 (-0.65*u3*(u1+a0dot) +2.6*F -0.3*u3*u5*u7-0.94*u3*u6*u5) -0.35*u2*u3]

    B =  [-3.1+12.8*u3^2 -2.6*u5;
          0.89-3.9*u3^2 -2.6*u5;
          0 0;
          0 2*u3;
          0 0;
          0 2.6;
          0 -4.3+1.5*u3^2]


    generate_distraction = ()  -> generate_distraction_from_gp(0.2 ;scale = 0.15, dims = 2)
    # We draw the control disturbations form a Gp.


    arguments= ( reltol = 1e-4, abstol = 1e-4,  init_noise = 0.0, init = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  clip_ctrl = 20, R = Diagonal([0.7, 0.08]), Q = Diagonal([1.0, 1, 10, 1, 1, 1, 1]), refstate = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    hp = ControlSystemDataSet( xsym, [c1,c2], A, B, timestep, arguments; hyp = hyps, hyp_vals= hypvalues, generate_distraction = generate_distraction)
    return Gaussian( hp, "Unicycle")

end




# We use the notation from Julia Differential Equations, therefore u corresponds to x
# And p are the hyper parameters including the controls.
