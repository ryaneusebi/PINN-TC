import tensorflow as tf

"""
Fundamental Equations of the Problem
"""

def Data_Equations(y, x, t, p, Y, neural_net):
   """The equations for matching neural net prediction and observed data"""
   u_data = Y[..., 0:1]
   v_data = Y[..., 1:2]
   h_data = Y[..., 2:3]
   nn_forward = neural_net(tf.concat([y,x,t,p],1))
   u_pred = nn_forward[..., 0:1]
   v_pred = nn_forward[..., 1:2]
   h_pred = nn_forward[..., 2:3]
   return u_data - u_pred, v_data - v_pred, h_data - h_pred

def Inverse_1stOrder_Equations_terms():
    
   ##########################
   # args: x_0,t_0,p_0,u_0,h_0,f,beta
   def inverse_1st_order_terms(y, x, t, p, args, neural_net=None, drop_mass_balance: bool = False):
      
      x_0,t_0,p_0,u_0,h_0,f,beta = args

      with tf.GradientTape(persistent=True) as tg:
         tg.watch(y)  # define the variable with respect to which you want to take derivative
         tg.watch(x)
         tg.watch(t)
         tg.watch(p)
         
         nn_forward = neural_net(tf.concat([y,x,t,p],1))
         u = nn_forward[..., 0:1]
         v = nn_forward[..., 1:2]
         h = nn_forward[..., 2:3]
         w = nn_forward[..., 3:4]
      
      
      u_t = tg.gradient(u, t)
      v_t = tg.gradient(v, t)
      u_x = tg.gradient(u, x)
      v_x = tg.gradient(v, x)
      u_y = tg.gradient(u, y)
      v_y = tg.gradient(v, y)
      h_x = tg.gradient(h, x)
      h_y = tg.gradient(h, y)
      u_p = tg.gradient(u, p)
      v_p = tg.gradient(v, p)
      w_p = tg.gradient(w, p)
      
      
      cor = f + beta*y*x_0
      # Momentum balance governing equations in horizontal direction
      ns_x = x_0/u_0/t_0*u_t + u*u_x + v*u_y + x_0/u_0/t_0*w*u_p - x_0/u_0*cor*v + (h_0/u_0**2)*h_x
      ns_y = x_0/u_0/t_0*v_t + u*v_x + v*v_y + x_0/u_0/t_0*w*v_p + x_0/u_0*cor*u + (h_0/u_0**2)*h_y

      nsx_terms = {
         'dt': x_0/u_0/t_0*u_t, 
         'advec': u*u_x + v*u_y + x_0/u_0/t_0*w*u_p, 
         'cor': -x_0/u_0*cor*v,
         'pres': (h_0/u_0**2)*h_x
      }

      nsy_terms = {
         'dt': x_0/u_0/t_0*v_t, 
         'advec': u*v_x + v*v_y + x_0/u_0/t_0*w*v_p, 
         'cor': x_0/u_0*cor*u,
         'pres': (h_0/u_0**2)*h_y
      }

      nsx_terms_advec = {
         'xadvec': u*u_x,
         'yadvec': v*u_y,
         'padvec': x_0/u_0/t_0*w*u_p
      }

      nsy_terms_advec = {
         'xadvec': u*v_x,
         'yadvec': v*v_y,
         'padvec': x_0/u_0/t_0*w*v_p
      }

      cont_terms = {
         'dudx': u_x,
         'dvdy': v_y,
         'dwdp': x_0/u_0/t_0*w_p
      }


      if drop_mass_balance:
         return nsx_terms, nsy_terms, nsx_terms_advec, nsy_terms_advec
      else:
         # mass balance governing equation
         cont = u_x + v_y + x_0/p_0*w_p
         return nsx_terms, nsy_terms, cont_terms, nsx_terms_advec, nsy_terms_advec

   return inverse_1st_order_terms


def Inverse_1stOrder_Equations():
    
   ##########################
   # args: x_0,t_0,p_0,u_0,h_0,f,beta
   def inverse_1st_order(y, x, t, p, args, neural_net=None, drop_mass_balance: bool = False):
      
      x_0,t_0,p_0,u_0,h_0,f,beta = args

      with tf.GradientTape(persistent=True) as tg:
         tg.watch(y)  # define the variable with respect to which you want to take derivative
         tg.watch(x)
         tg.watch(t)
         tg.watch(p)
         
         nn_forward = neural_net(tf.concat([y,x,t,p],1))
         u = nn_forward[..., 0:1]
         v = nn_forward[..., 1:2]
         h = nn_forward[..., 2:3]
         w = nn_forward[..., 3:4]      
      
      u_t = tg.gradient(u, t)
      v_t = tg.gradient(v, t)
      u_x = tg.gradient(u, x)
      v_x = tg.gradient(v, x)
      u_y = tg.gradient(u, y)
      v_y = tg.gradient(v, y)
      h_x = tg.gradient(h, x)
      h_y = tg.gradient(h, y)
      u_p = tg.gradient(u, p)
      v_p = tg.gradient(v, p)
      w_p = tg.gradient(w, p)
      
      
      cor = f + beta*y*x_0
      # Momentum balance governing equations in horizontal direction
      ns_x = x_0/u_0/t_0*u_t + u*u_x + v*u_y + x_0/u_0/t_0*w*u_p - x_0/u_0*cor*v + (h_0/u_0**2)*h_x
      ns_y = x_0/u_0/t_0*v_t + u*v_x + v*v_y + x_0/u_0/t_0*w*v_p + x_0/u_0*cor*u + (h_0/u_0**2)*h_y

      if drop_mass_balance:
         return ns_x, ns_y
      else:
         # mass balance governing equation
         cont = u_x + v_y + x_0/u_0/t_0*w_p
         return ns_x, ns_y, cont

   return inverse_1st_order
                