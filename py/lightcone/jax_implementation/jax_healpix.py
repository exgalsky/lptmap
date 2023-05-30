from functools import partial
import jax
import jax.numpy as jnp 

@jax.jit
def imodulo64(v1, v2):
    v = jnp.int64(v1%v2)
    return jnp.where(v >= 0, v, v + v2)

@jax.jit
def fmodulo(v1, v2):
    temp = jnp.mod(v1 , v2) + v2
    return jnp.where(v1 >= 0., jnp.where(v1 < v2, v1, jnp.mod(v1,v2)), jnp.where(temp == 0., 0., temp))

# @partial(jax.jit, static_argnames=['nside'])
# def zt2pix_ring_eq(nside, z, tt):
#     temp1 = nside * (tt + 0.5)
#     temp2 = nside * z * 0.75 

#     jp = jnp.int64(temp1 - temp2)
#     jm = jnp.int64(temp1 + temp2)
#     ir = nside + 1 + jp - jm

#     kshift = 1 - (ir & 1)
#     ip = imodulo64(jnp.int64((jp + jm - nside + kshift + 1) // 2), 4*nside)

#     return jnp.int64(nside * (nside - 1) * 2 + (ir - 1) * 4*nside + ip)

# @partial(jax.jit, static_argnames=['nside'])
# def zt2pix_ring_pol(nside, z, za, tt, s):
#     tp = tt - jnp.int16(tt)
#     temp = jnp.where(s > -2., nside * s / jnp.sqrt((1. + za) / 3.), nside * jnp.sqrt(3. * ( 1. - za)))

#     jp = jnp.int64(tp * temp)
#     jm = jnp.int64((1. - tp) * temp)

#     ir = jnp.int64(jp + jm + 1)
#     ip = imodulo64(jnp.int64(tt * ir), 4*ir)

#     return jnp.int64(jnp.where(z > 0, 2 * ir * (ir - 1) + ip, 12 * nside * nside - 2 * ir * (ir + 1) + ip))


@partial(jax.jit, static_argnames=['nside'])
def ang2pix_ring_zphi(nside, z, s, phi):
    za = jnp.abs(z)
    print(z.shape, s.shape, phi.shape)
    tt = jax.vmap(fmodulo, in_axes=(0, None), out_axes=0)(phi, 2*jnp.pi) * (2. / jnp.pi)

    def zt2pix_ring_eq(nside, z, tt):
        temp1 = nside * (tt + 0.5)
        temp2 = nside * z * 0.75 

        jp = jnp.int64(temp1 - temp2)
        jm = jnp.int64(temp1 + temp2)
        ir = nside + 1 + jp - jm

        kshift = 1 - (ir & 1)
        ip = imodulo64(jnp.int64((jp + jm - nside + kshift + 1) // 2), 4*nside)

        return jnp.int64(nside * (nside - 1) * 2 + (ir - 1) * 4*nside + ip)
    
    def zt2pix_ring_pol(nside, z, za, tt, s):
        tp = tt - jnp.int16(tt)
        temp = jnp.where(s > -2., nside * s / jnp.sqrt((1. + za) / 3.), nside * jnp.sqrt(3. * ( 1. - za)))

        jp = jnp.int64(tp * temp)
        jm = jnp.int64((1. - tp) * temp)

        ir = jnp.int64(jp + jm + 1)
        ip = imodulo64(jnp.int64(tt * ir), 4*ir)

        return jnp.int64(jnp.where(z > 0, 2 * ir * (ir - 1) + ip, 12 * nside * nside - 2 * ir * (ir + 1) + ip))

    return jnp.where(za <= (2./3.), jax.vmap(zt2pix_ring_eq, in_axes=(None, 0, 0), out_axes=0)(nside, z, tt), 
                     jax.vmap(zt2pix_ring_pol, in_axes=(None, 0, 0, 0, 0), out_axes=0)(nside, z, za, tt, s))


@partial(jax.jit, static_argnames=['nside'])
def vec2pix(nside, x, y, z):
    
    vlen = jnp.sqrt(x*x + y*y + z*z)
    cth = z / vlen
    sth = jnp.where(jnp.abs(cth) > 0.99, jnp.sqrt(x*x + y*y) / vlen, -5.)

    return jax.vmap(ang2pix_ring_zphi, in_axes=(None, 0, 0, 0), out_axes=0)(nside, cth, sth, jnp.arctan2(y, x))


