import streamlit as st

# Configure page
st.set_page_config(
    page_title="RF Reflector Simulator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import io

# Simple CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def linspace(start, stop, num):
    """Generate linearly spaced array"""
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]

def dot(a, b):
    """Dot product of two 3D vectors"""
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def subtract(a, b):
    """Subtract two 3D vectors"""
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def add(a, b):
    """Add two 3D vectors"""
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]

def scale(v, s):
    """Scale a 3D vector by scalar"""
    return [v[0]*s, v[1]*s, v[2]*s]

def norm(v):
    """Calculate norm of 3D vector"""
    return np.sqrt(dot(v, v))

def normalize(v):
    """Normalize a 3D vector"""
    n = norm(v)
    return [0, 0, 0] if n == 0 else [v[0]/n, v[1]/n, v[2]/n]

def cross(a, b):
    """Cross product of two 3D vectors"""
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]

@st.cache_data
def generate_paraboloid_surface(focal, hole_radius, dish_radius):
    """Generate paraboloid surface data for 3D plotting"""
    theta_steps = 60
    r_steps = 50
    
    theta_vals = linspace(0, 2*np.pi, theta_steps)
    r_vals = linspace(hole_radius, dish_radius, r_steps)
    
    X, Y, Z = [], [], []
    
    for theta in theta_vals:
        row_x, row_y, row_z = [], [], []
        for r in r_vals:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = ((r - hole_radius) ** 2) / (4 * focal)
            
            row_x.append(x)
            row_y.append(y)
            row_z.append(z)
        
        X.append(row_x)
        Y.append(row_y)
        Z.append(row_z)
    
    return np.array(X), np.array(Y), np.array(Z)

@st.cache_data
def compute_catacaustic_points(angle_deg, focal, hole_radius, dish_radius):
    """Compute the 2f catacaustic points (envelope of reflected rays)"""
    points = []
    theta_steps = 90
    r_steps = 50
    
    theta_vals = linspace(0, 2*np.pi, theta_steps)
    
    # Handle special case for normal incidence with no hole
    if abs(angle_deg - 90) < 1e-6 and hole_radius == 0:
        r_vals = [0]
    else:
        r_vals = linspace(hole_radius, dish_radius, r_steps)
    
    alpha = (90 - angle_deg) * np.pi / 180  # incidence angle from horizontal
    D = [np.sin(alpha), 0, -np.cos(alpha)]  # incident direction vector
    
    for theta in theta_vals:
        for r in r_vals:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = ((r - hole_radius) ** 2) / (4 * focal)
            P = [x, y, z]  # point on dish
            
            # Find intersection with rotated focal plane
            t_param = dot(D, P) + focal
            I = subtract(P, scale(D, t_param))
            d_inc = norm(subtract(P, I))  # incident distance
            d_ref = 2 * focal - d_inc     # remaining distance for 2f total
            
            # Compute surface normal at P
            r_p = np.sqrt(x*x + y*y)
            if r_p == 0:
                N = [0, 0, 1]
            else:
                dz_dr = (r_p - hole_radius) / (2 * focal)
                N = normalize([-dz_dr * (x / r_p), -dz_dr * (y / r_p), 1])
            
            # Only consider illuminated points
            if dot(D, N) >= 0:
                continue
            
            # Reflect D across N
            d_dot_n = dot(D, N)
            R_vec = normalize([
                D[0] - 2 * d_dot_n * N[0],
                D[1] - 2 * d_dot_n * N[1],
                D[2] - 2 * d_dot_n * N[2]
            ])
            
            # Final reflected point
            F = [P[0] + d_ref * R_vec[0],
                 P[1] + d_ref * R_vec[1],
                 P[2] + d_ref * R_vec[2]]
            
            points.append(F)
    
    return points

def intersect_ray_with_paraboloid(P0, D, focal, hole_radius):
    """Find intersection of ray with paraboloid using bisection method"""
    def f(t):
        x = P0[0] + t * D[0]
        y = P0[1] + t * D[1]
        z = P0[2] + t * D[2]
        r = np.sqrt(x*x + y*y)
        return z - ((r - hole_radius) ** 2) / (4 * focal)
    
    t_low = 0
    t_high = 2 * focal
    
    # Expand t_high until sign change
    while f(t_low) * f(t_high) > 0 and t_high < 100 * focal:
        t_high *= 2
    
    # Bisection method
    tol = 1e-8
    for _ in range(60):
        t_mid = (t_low + t_high) / 2
        f_mid = f(t_mid)
        if abs(f_mid) < tol:
            break
        if f(t_low) * f_mid < 0:
            t_high = t_mid
        else:
            t_low = t_mid
    
    return add(P0, scale(D, t_mid))

def compute_ray_path(angle_deg, focal, hole_radius, ray_r, ray_theta_deg):
    """Compute single ray path for animation"""
    alpha = (90 - angle_deg) * np.pi / 180
    D = [np.sin(alpha), 0, -np.cos(alpha)]
    z_source = focal + 0.5
    
    t_rad = ray_theta_deg * np.pi / 180
    I0 = [ray_r * np.cos(t_rad), ray_r * np.sin(t_rad), z_source]
    
    # Find intersection with paraboloid
    P = intersect_ray_with_paraboloid(I0, D, focal, hole_radius)
    
    # Intersection with rotated focal plane
    t_param = dot(D, P) + focal
    I = subtract(P, scale(D, t_param))
    d_inc = norm(subtract(P, I))
    d_ref = 2 * focal - d_inc
    
    # Surface normal at P
    r_p = np.sqrt(P[0]*P[0] + P[1]*P[1])
    if r_p == 0:
        N = [0, 0, 1]
    else:
        dz_dr = (r_p - hole_radius) / (2 * focal)
        N = normalize([-dz_dr * (P[0]/r_p), -dz_dr * (P[1]/r_p), 1])
    
    # Reflected direction
    d_dot_n = dot(D, N)
    R_vec = normalize([
        D[0] - 2 * d_dot_n * N[0],
        D[1] - 2 * d_dot_n * N[1],
        D[2] - 2 * d_dot_n * N[2]
    ])
    
    # End point of reflected ray
    R = [P[0] + d_ref * R_vec[0],
         P[1] + d_ref * R_vec[1],
         P[2] + d_ref * R_vec[2]]
    
    return {
        'I': I, 'P': P, 'R': R,
        'd_inc': d_inc, 'd_ref': d_ref
    }

def generate_incident_rays(focal, angle_deg, dish_radius, hole_radius, grid_res):
    """Generate incident ray array for visualization"""
    rays = []
    alpha = (90 - angle_deg) * np.pi / 180
    D = [np.sin(alpha), 0, -np.cos(alpha)]
    z_source = focal + 0.5
    
    coords = linspace(-dish_radius, dish_radius, grid_res)
    
    for x in coords:
        for y in coords:
            P0 = [x, y, z_source]
            Q = intersect_ray_with_paraboloid(P0, D, focal, hole_radius)
            rays.append({'start': P0, 'end': Q})
    
    return rays

def generate_focal_plane(focal, dish_radius):
    """Generate focal plane mesh"""
    plane_z = focal
    range_val = dish_radius
    
    xs = linspace(-range_val, range_val, 20)
    ys = linspace(-range_val, range_val, 20)
    
    X, Y, Z = [], [], []
    for x in xs:
        row_x, row_y, row_z = [], [], []
        for y in ys:
            row_x.append(x)
            row_y.append(y)
            row_z.append(plane_z)
        X.append(row_x)
        Y.append(row_y)
        Z.append(row_z)
    
    return np.array(X), np.array(Y), np.array(Z)

def generate_rotated_plane(focal, angle_deg, dish_radius):
    """Generate rotated focal plane perpendicular to incident direction"""
    alpha = (90 - angle_deg) * np.pi / 180
    D = [np.sin(alpha), 0, -np.cos(alpha)]
    P0 = scale(D, -focal)  # plane center
    
    # Two in-plane axes
    up = [0, 1, 0]
    v = normalize(cross(D, up))
    
    N = 20
    u_vals = linspace(-dish_radius, dish_radius, N)
    v_vals = linspace(-dish_radius, dish_radius, N)
    
    X, Y, Z = [], [], []
    for u in u_vals:
        row_x, row_y, row_z = [], [], []
        for v_val in v_vals:
            pt = add(add(P0, scale(up, u)), scale(v, v_val))
            row_x.append(pt[0])
            row_y.append(pt[1])
            row_z.append(pt[2])
        X.append(row_x)
        Y.append(row_y)
        Z.append(row_z)
    
    return np.array(X), np.array(Y), np.array(Z)

def create_3d_plot(angle_deg, hole_radius, focal, dish_diameter, 
                   show_incident, grid_res, show_focal_plane, show_rotated_plane, single_ray_data=None):
    """Create the main 3D visualization"""
    dish_radius = dish_diameter / 2
    
    # Generate surface data
    X_surf, Y_surf, Z_surf = generate_paraboloid_surface(focal, hole_radius, dish_radius)
    
    # Generate catacaustic points
    cat_points = compute_catacaustic_points(angle_deg, focal, hole_radius, dish_radius)
    
    # Create figure
    fig = go.Figure()
    
    # Add paraboloid surface
    fig.add_trace(go.Surface(
        x=X_surf, y=Y_surf, z=Z_surf,
        colorscale='Viridis',
        opacity=0.7,
        showscale=False,
        name='Parabolic Dish'
    ))
    
    # Add catacaustic points
    if cat_points:
        cat_x = [p[0] for p in cat_points]
        cat_y = [p[1] for p in cat_points]
        cat_z = [p[2] for p in cat_points]
        
        fig.add_trace(go.Scatter3d(
            x=cat_x, y=cat_y, z=cat_z,
            mode='markers',
            marker=dict(size=2, color='red'),
            name='2f Catacaustic'
        ))
    
    # Add focal plane
    if show_focal_plane:
        X_focal, Y_focal, Z_focal = generate_focal_plane(focal, dish_radius)
        fig.add_trace(go.Surface(
            x=X_focal, y=Y_focal, z=Z_focal,
            opacity=0.25,
            colorscale=[[0, 'cyan'], [1, 'cyan']],
            showscale=False,
            name='Focal Plane (z = f)'
        ))
    
    # Add rotated plane
    if show_rotated_plane:
        X_rot, Y_rot, Z_rot = generate_rotated_plane(focal, angle_deg, dish_radius)
        fig.add_trace(go.Surface(
            x=X_rot, y=Y_rot, z=Z_rot,
            opacity=0.25,
            colorscale=[[0, 'magenta'], [1, 'magenta']],
            showscale=False,
            name='Rotated Focal Plane'
        ))
    
    # Add incident rays
    if show_incident:
        incident_rays = generate_incident_rays(focal, angle_deg, dish_radius, hole_radius, grid_res)
        for ray in incident_rays:
            fig.add_trace(go.Scatter3d(
                x=[ray['start'][0], ray['end'][0]],
                y=[ray['start'][1], ray['end'][1]],
                z=[ray['start'][2], ray['end'][2]],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
    
    # Add single ray if provided
    if single_ray_data:
        ray = single_ray_data
        fig.add_trace(go.Scatter3d(
            x=[ray['I'][0], ray['P'][0], ray['R'][0]],
            y=[ray['I'][1], ray['P'][1], ray['R'][1]],
            z=[ray['I'][2], ray['P'][2], ray['R'][2]],
            mode='lines+markers',
            line=dict(color='orange', width=6),
            marker=dict(size=6, color='orange'),
            name='Single Ray'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Paraboloid Reflector (f={focal:.3f} m) – RF Rays & 2f Focal Envelope',
        scene=dict(
            aspectmode='data',
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            bgcolor='rgba(0,0,0,0)'
        ),
        height=700,
        template='plotly_dark'
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📡 RF Paraboloid Reflector Simulation</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Reflector Parameters")
    
    # Main parameters
    incident_angle = st.sidebar.slider(
        "Incident Angle (degrees):",
        min_value=0.0,
        max_value=90.0,
        value=90.0,
        step=0.1,
        help="Angle of incoming RF waves"
    )
    
    hole_radius = st.sidebar.number_input(
        "Hole Radius (m):",
        min_value=0.0,
        max_value=0.5,
        value=0.01524,
        step=0.005,
        format="%.5f",
        help="Radius of central hole in dish"
    )
    
    focal_length = st.sidebar.number_input(
        "Focal Length (m):",
        min_value=0.05,
        max_value=1.0,
        value=0.3299,
        step=0.005,
        format="%.4f",
        help="Distance from dish vertex to focus"
    )
    
    dish_diameter = st.sidebar.number_input(
        "Dish Diameter (m):",
        min_value=0.5,
        max_value=5.0,
        value=1.35,
        step=0.05,
        format="%.2f",
        help="Total diameter of parabolic dish"
    )
    
    # Visualization options
    st.sidebar.header("🔍 Visualization Options")
    
    show_incident = st.sidebar.checkbox(
        "Show Incident Rays",
        value=True,
        help="Display incoming ray grid"
    )
    
    if show_incident:
        grid_res = st.sidebar.slider(
            "Grid Resolution:",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of rays per side"
        )
    else:
        grid_res = 5
    
    show_focal_plane = st.sidebar.checkbox(
        "Show Focal Plane (z = f)",
        value=True,
        help="Display horizontal focal plane"
    )
    
    show_rotated_plane = st.sidebar.checkbox(
        "Show Rotated Focal Plane",
        value=True,
        help="Display plane perpendicular to incident rays"
    )
    
    # Single ray analysis
    st.sidebar.header("🔬 Single Ray Analysis")
    
    analyze_ray = st.sidebar.checkbox(
        "Analyze Single Ray",
        value=False,
        help="Show detailed path of one ray"
    )
    
    if analyze_ray:
        ray_radius = st.sidebar.slider(
            "Ray Radius (m):",
            min_value=hole_radius,
            max_value=dish_diameter/2,
            value=dish_diameter/4,
            step=0.01,
            help="Radial position of ray on dish"
        )
        
        ray_theta = st.sidebar.slider(
            "Ray Angle (degrees):",
            min_value=0.0,
            max_value=360.0,
            value=0.0,
            step=1.0,
            help="Angular position of ray"
        )
    
    # Info section
    with st.sidebar.expander("ℹ️ About This Simulation"):
        st.write("""
        **RF Paraboloid Reflector:**
        - Models electromagnetic wave reflection
        - Red points show 2f catacaustic envelope
        - All reflected rays have path length = 2×focal length
        - Used in satellite dishes, radio telescopes
        
        **Colors:**
        - 🟦 Blue: Incident rays
        - 🔴 Red: 2f focal envelope points
        - 🟠 Orange: Single ray path
        - 🟢 Cyan: Focal plane
        - 🟣 Magenta: Rotated focal plane
        """)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.header("📊 Analysis")
        
        # Key metrics
        st.metric("Focal Length", f"{focal_length:.4f} m")
        st.metric("F-Number", f"{focal_length/(dish_diameter):.2f}")
        st.metric("Dish Area", f"{np.pi * (dish_diameter/2)**2:.3f} m²")
        
        if hole_radius > 0:
            blocked_area = np.pi * hole_radius**2
            efficiency = 1 - (blocked_area / (np.pi * (dish_diameter/2)**2))
            st.metric("Aperture Efficiency", f"{efficiency:.1%}")
        
        # Single ray analysis
        if analyze_ray:
            st.subheader("🔬 Ray Path Analysis")
            ray_data = compute_ray_path(incident_angle, focal_length, hole_radius, ray_radius, ray_theta)
            
            st.write("**Path Distances:**")
            st.write(f"Incident: {ray_data['d_inc']:.4f} m")
            st.write(f"Reflected: {ray_data['d_ref']:.4f} m")
            st.write(f"Total: {ray_data['d_inc'] + ray_data['d_ref']:.4f} m")
            st.write(f"2×Focal: {2*focal_length:.4f} m")
            
            # Verify 2f property
            total_path = ray_data['d_inc'] + ray_data['d_ref']
            error = abs(total_path - 2*focal_length)
            if error < 0.001:
                st.success("✅ Path length = 2f (verified)")
            else:
                st.warning(f"⚠️ Path error: {error:.6f} m")
    
    with col1:
        # Generate visualization
        with st.spinner("🔄 Computing RF simulation..."):
            single_ray_data = None
            if analyze_ray:
                single_ray_data = compute_ray_path(incident_angle, focal_length, hole_radius, ray_radius, ray_theta)
            
            fig = create_3d_plot(
                incident_angle, hole_radius, focal_length, dish_diameter,
                show_incident, grid_res, show_focal_plane, show_rotated_plane,
                single_ray_data
            )
        
        # Display plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.subheader("💾 Export Data")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("📊 Export Catacaustic Points"):
                cat_points = compute_catacaustic_points(incident_angle, focal_length, hole_radius, dish_diameter/2)
                if cat_points:
                    df = pd.DataFrame(cat_points, columns=['X', 'Y', 'Z'])
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="2f_catacaustic_points.csv",
                        mime="text/csv"
                    )
                    
                    st.success(f"✅ Generated {len(cat_points)} points")
                else:
                    st.warning("No catacaustic points generated")
        
        with col_b:
            if analyze_ray and st.button("📈 Export Ray Data"):
                ray_data = compute_ray_path(incident_angle, focal_length, hole_radius, ray_radius, ray_theta)
                
                ray_df = pd.DataFrame({
                    'Point': ['Incident', 'Reflection', 'End'],
                    'X': [ray_data['I'][0], ray_data['P'][0], ray_data['R'][0]],
                    'Y': [ray_data['I'][1], ray_data['P'][1], ray_data['R'][1]],
                    'Z': [ray_data['I'][2], ray_data['P'][2], ray_data['R'][2]]
                })
                
                csv = ray_df.to_csv(index=False)
                st.download_button(
                    label="Download Ray CSV",
                    data=csv,
                    file_name="single_ray_path.csv",
                    mime="text/csv"
                )

# Run the app
main()
