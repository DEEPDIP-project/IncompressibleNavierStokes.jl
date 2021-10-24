"""
    update_rtp!(rtp, setup, V, p, t)

Update real time plot.
"""
function update_rtp!(rtp, setup, V, p, t)
    @unpack Nx, Ny, Npx, Npy = setup.grid
    @unpack rtp_type = setup.visualization
    if rtp_type == "velocity"
        up, vp, qp = get_velocity(V, t, setup)
        rtp.vel[] = qp
    elseif rtp_type == "vorticity"
        rtp.ω[] = vorticity!(rtp.ω[], V, t, setup)
    elseif rtp_type == "streamfunction"
        rtp.ψ[] = get_streamfunction(V, t, setup)
    elseif rtp_type == "pressure"
        rtp.pres[] = reshape(p, Npx, Npy)
    end
    # sleep(1 / rtp.fps)
end
