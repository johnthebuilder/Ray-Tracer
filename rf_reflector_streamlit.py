import streamlit as st

# Configure page
st.set_page_config(
    page_title="RF Reflector Simulator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
import pandas as pd
import io
import time
import plotly.graph_objects as go
from PIL import Image

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
    """Generate paraboloid surface data - opening upward with focus at top"""
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
            # Paraboloid opening upward: z = r²/(4f), focus at (0,0,f) above
            z = (r ** 2) / (4 * focal)
            
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
    r_vals = linspace(hole_radius, dish_radius, r_steps)
    
    # Incident direction - from above
    alpha = (90 - angle_deg) * np.pi / 180  # angle from horizontal
    D = [np.sin(alpha), 0, -np.cos(alpha)]  # incident direction (downward)
    
    for theta in theta_vals:
        for r in r_vals:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            # Point on upward-opening paraboloid
            z = (r ** 2) / (4 * focal)
            P = [x, y, z]  # point on dish
            
            # Surface normal at P (pointing outward/upward)
            if r == 0:
                N = [0, 0, 1]  # normal at vertex points up
            else:
                # For z = r²/(4f), dz/dx = x/(2f), dz/dy = y/(2f)
                # Normal = (-dz/dx, -dz/dy, 1) = (-x/(2f), -y/(2f), 1)
                N = normalize([-x/(2*focal), -y/(2*focal), 1])
            
            # Only consider illuminated points (incident ray hits surface)
            if dot(D, N) >= 0:
                continue
            
            # Reflect incident direction across normal
            d_dot_n = dot(D, N)
            R_vec = normalize([
                D[0] - 2 * d_dot_n * N[0],
                D[1] - 2 * d_dot_n * N[1],
                D[2] - 2 * d_dot_n * N[2]
            ])
            
            # For 2f total path length
            # Distance from focal plane to dish point
            t_param = dot(D, P) + focal
            I = subtract(P, scale(D, t_param))
            d_inc = norm(subtract(P, I))
            d_ref = 2 * focal - d_inc
            
            # End point of reflected ray
            if d_ref > 0:
                F = [P[0] + d_ref * R_vec[0],
                     P[1] + d_ref * R_vec[1],
                     P[2] + d_ref * R_vec[2]]
                points.append(F)
    
    return points

def get_color_for_angle(angle, min_angle, max_angle):
    """Get a distinct color for each angle using a color gradient"""
    # Normalize angle to 0-1 range
    normalized = (angle - min_angle) / (max_angle - min_angle) if max_angle != min_angle else 0.5
    
    # Use a color scale from blue (cold/low angles) to red (hot/high angles)
    # You can modify this to use other color schemes
    colors = [
        '#0000FF',  # Blue
        '#0080FF',  # Light Blue
        '#00FFFF',  # Cyan
        '#00FF80',  # Green-Cyan
        '#00FF00',  # Green
        '#80FF00',  # Yellow-Green
        '#FFFF00',  # Yellow
        '#FF8000',  # Orange
        '#FF0000',  # Red
        '#FF0080',  # Pink-Red
    ]
    
    # Get color index
    color_index = int(normalized * (len(colors) - 1))
    color_index = max(0, min(color_index, len(colors) - 1))
    
    return colors[color_index]

def intersect_ray_with_paraboloid(P0, D, focal, hole_radius):
    """Find intersection of ray with paraboloid using bisection method"""
    def f(t):
        x = P0[0] + t * D[0]
        y = P0[1] + t * D[1]
        z = P0[2] + t * D[2]
        r = np.sqrt(x*x + y*y)
        return z - (r ** 2) / (4 * focal)
    
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
    """Compute single ray path for analysis"""
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
        N = normalize([-P[0]/(2*focal), -P[1]/(2*focal), 1])
    
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
    """Generate focal plane mesh at z = focal"""
    xs = linspace(-dish_radius, dish_radius, 20)
    ys = linspace(-dish_radius, dish_radius, 20)
    
    X, Y, Z = [], [], []
    for x in xs:
        row_x, row_y, row_z = [], [], []
        for y in ys:
            row_x.append(x)
            row_y.append(y)
            row_z.append(focal)
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

def get_catacaustic_bounds(cat_points, focal, hole_radius):
    """Get bounds for zoomed view of catacaustic region"""
    if not cat_points:
        return {
            'x': [-0.1, 0.1], 'y': [-0.1, 0.1], 'z': [focal-0.1, focal+0.1]
        }
    
    cat_x = [p[0] for p in cat_points]
    cat_y = [p[1] for p in cat_points]
    cat_z = [p[2] for p in cat_points]
    
    # Get bounds with some padding
    x_margin = max(0.05, (max(cat_x) - min(cat_x)) * 0.2)
    y_margin = max(0.05, (max(cat_y) - min(cat_y)) * 0.2)
    z_margin = max(0.05, (max(cat_z) - min(cat_z)) * 0.2)
    
    # If there's a hole, ensure we show the ring structure
    if hole_radius > 0:
        x_margin = max(x_margin, hole_radius * 2)
        y_margin = max(y_margin, hole_radius * 2)
    
    return {
        'x': [min(cat_x) - x_margin, max(cat_x) + x_margin],
        'y': [min(cat_y) - y_margin, max(cat_y) + y_margin],
        'z': [min(cat_z) - z_margin, max(cat_z) + z_margin]
    }

def create_plotly_visualization(angle_deg, hole_radius, focal, dish_diameter, 
                              show_incident, grid_res, show_focal_plane, 
                              show_rotated_plane, single_ray_data=None, 
                              zoom_to_catacaustic=False):
    """Create Plotly 3D visualization"""
    dish_radius = dish_diameter / 2
    
    # Generate surface data
    X_surf, Y_surf, Z_surf = generate_paraboloid_surface(focal, hole_radius, dish_radius)
    
    # Generate catacaustic points
    cat_points = compute_catacaustic_points(angle_deg, focal, hole_radius, dish_radius)
    
    # Create figure
    fig = go.Figure()
    
    # Add paraboloid surface (reduced opacity for animation focus)
    opacity = 0.3 if zoom_to_catacaustic else 0.7
    fig.add_trace(go.Surface(
        x=X_surf, y=Y_surf, z=Z_surf,
        colorscale='Viridis',
        opacity=opacity,
        showscale=False,
        name='Parabolic Dish'
    ))
    
    # Add focus point
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[focal],
        mode='markers',
        marker=dict(size=10, color='yellow', symbol='diamond'),
        name='Focus Point'
    ))
    
    # Add focal plane (reduced opacity for animation)
    if show_focal_plane:
        X_focal, Y_focal, Z_focal = generate_focal_plane(focal, dish_radius)
        focal_opacity = 0.1 if zoom_to_catacaustic else 0.25
        fig.add_trace(go.Surface(
            x=X_focal, y=Y_focal, z=Z_focal,
            opacity=focal_opacity,
            colorscale=[[0, 'cyan'], [1, 'cyan']],
            showscale=False,
            name='Focal Plane (z = f)'
        ))
    
    # Add rotated plane (reduced opacity for animation)
    if show_rotated_plane:
        X_rot, Y_rot, Z_rot = generate_rotated_plane(focal, angle_deg, dish_radius)
        rot_opacity = 0.1 if zoom_to_catacaustic else 0.25
        fig.add_trace(go.Surface(
            x=X_rot, y=Y_rot, z=Z_rot,
            opacity=rot_opacity,
            colorscale=[[0, 'magenta'], [1, 'magenta']],
            showscale=False,
            name='Rotated Focal Plane'
        ))
    
    # Add incident rays (fewer for animation)
    if show_incident and not zoom_to_catacaustic:
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
    
    # Add catacaustic points (ultra-small marker sizes for finest detail)
    if cat_points:
        cat_x = [p[0] for p in cat_points]
        cat_y = [p[1] for p in cat_points]
        cat_z = [p[2] for p in cat_points]
        
        # Ultra-small marker sizes for finest field visualization
        marker_size = 0.6 if zoom_to_catacaustic else 0.4  # Even smaller: 1.0/0.8 → 0.6/0.4
        fig.add_trace(go.Scatter3d(
            x=cat_x, y=cat_y, z=cat_z,
            mode='markers',
            marker=dict(size=marker_size, color='red'),
            name='2f Catacaustic'
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
    
    # Set camera and bounds with enhanced viewpoint for catacaustic visualization
    if zoom_to_catacaustic:
        bounds = get_catacaustic_bounds(cat_points, focal, hole_radius)
        scene_dict = dict(
            aspectmode='cube',
            xaxis=dict(title='X (m)', range=bounds['x']),
            yaxis=dict(title='Y (m)', range=bounds['y']),
            zaxis=dict(title='Z (m)', range=bounds['z']),
            # Enhanced viewing angle rotated 45° around z-axis for better catacaustic field visualization
            camera=dict(eye=dict(x=2.5, y=0.5, z=1.5))  # Rotated ~45° from (1.8,1.8,1.5)
        )
        title_suffix = " - Zoomed to Catacaustic Focus"
    else:
        scene_dict = dict(
            aspectmode='data',
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            # Optimal viewing angle rotated 45° around z-axis for overall visualization  
            camera=dict(eye=dict(x=2.1, y=0.4, z=1.5))  # Rotated ~45° from (1.5,1.5,1.5)
        )
        title_suffix = ""
    
    # Update layout
    fig.update_layout(
        title=f'RF Paraboloid Reflector (f={focal:.3f} m, θ={angle_deg:.1f}°){title_suffix}',
        scene=scene_dict,
        height=700,
        template='plotly_dark',
        showlegend=not zoom_to_catacaustic  # Hide legend for cleaner animation
    )
    
    return fig

def create_multi_angle_visualization(angle_values, focal_length, hole_radius, dish_diameter, 
                                   show_focal_plane=True, show_rotated_plane=False):
    """Create visualization showing catacaustic points for all angles with distinct colors"""
    dish_radius = dish_diameter / 2
    
    # Generate surface data
    X_surf, Y_surf, Z_surf = generate_paraboloid_surface(focal_length, hole_radius, dish_radius)
    
    # Create figure
    fig = go.Figure()
    
    # Add paraboloid surface with good visibility
    fig.add_trace(go.Surface(
        x=X_surf, y=Y_surf, z=Z_surf,
        colorscale='Viridis',
        opacity=0.6,
        showscale=False,
        name='Parabolic Dish'
    ))
    
    # Add focus point
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[focal_length],
        mode='markers',
        marker=dict(size=12, color='yellow', symbol='diamond'),
        name='Focus Point'
    ))
    
    # Add focal plane
    if show_focal_plane:
        X_focal, Y_focal, Z_focal = generate_focal_plane(focal_length, dish_radius)
        fig.add_trace(go.Surface(
            x=X_focal, y=Y_focal, z=Z_focal,
            opacity=0.15,
            colorscale=[[0, 'cyan'], [1, 'cyan']],
            showscale=False,
            name='Focal Plane (z = f)'
        ))
    
    # Add catacaustic points for each angle with distinct colors (much smaller sizes)
    min_angle = min(angle_values)
    max_angle = max(angle_values)
    
    for i, angle in enumerate(angle_values):
        cat_points = compute_catacaustic_points(angle, focal_length, hole_radius, dish_radius)
        
        if cat_points:
            cat_x = [p[0] for p in cat_points]
            cat_y = [p[1] for p in cat_points]
            cat_z = [p[2] for p in cat_points]
            
            # Get distinct color for this angle
            color = get_color_for_angle(angle, min_angle, max_angle)
            
            fig.add_trace(go.Scatter3d(
                x=cat_x, y=cat_y, z=cat_z,
                mode='markers',
                marker=dict(size=0.5, color=color),  # Ultra-small: reduced from 0.8 to 0.5
                name=f'θ={angle:.1f}°'
            ))
    
    # Set scene properties for full dish view with 45° rotated viewpoint
    scene_dict = dict(
        aspectmode='data',
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        # Rotated 45° around z-axis from standard (1.5,1.5,1.5) viewpoint
        camera=dict(eye=dict(x=2.1, y=0.4, z=1.5))
    )
    
    # Update layout
    fig.update_layout(
        title=f'RF Paraboloid Multi-Angle Catacaustic (f={focal_length:.3f} m)',
        scene=scene_dict,
        height=700,
        template='plotly_dark',
        showlegend=True
    )
    
    return fig

def create_animation_gif(angle_values, focal_length, hole_radius, dish_diameter):
    """Create enhanced GIF animation showing catacaustic field collapse with improved visualization"""
    try:
        # Check if kaleido is available for image export
        import kaleido
    except ImportError:
        st.error("❌ Kaleido library not available for GIF export. Install with: pip install kaleido")
        return None
    
    images = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        dish_radius = dish_diameter / 2
        min_angle = min(angle_values)
        max_angle = max(angle_values)
        
        # Generate surface data once (static throughout animation)
        X_surf, Y_surf, Z_surf = generate_paraboloid_surface(focal_length, hole_radius, dish_radius)
        
        # Get all catacaustic points first to determine optimal bounds
        all_cat_points = []
        for angle in angle_values:
            cat_points = compute_catacaustic_points(angle, focal_length, hole_radius, dish_radius)
            all_cat_points.extend(cat_points)
        
        # Calculate enhanced bounds for better catacaustic visualization
        if all_cat_points:
            all_x = [p[0] for p in all_cat_points]
            all_y = [p[1] for p in all_cat_points]
            all_z = [p[2] for p in all_cat_points]
            
            # Tighter bounds focused on catacaustic region
            x_range = [min(all_x) * 1.3, max(all_x) * 1.3]
            y_range = [min(all_y) * 1.3, max(all_y) * 1.3]
            z_range = [min(all_z) * 0.8, max(all_z) * 1.2]
        else:
            # Fallback bounds
            x_range = [-0.2, 0.2]
            y_range = [-0.2, 0.2]
            z_range = [focal_length * 0.8, focal_length * 1.2]
        
        for i, angle in enumerate(angle_values):
            status_text.text(f"Generating enhanced frame {i+1}/{len(angle_values)} (θ={angle:.1f}°)")
            
            # Create figure for this frame
            fig = go.Figure()
            
            # Add paraboloid surface (static, reduced opacity for focus on points)
            fig.add_trace(go.Surface(
                x=X_surf, y=Y_surf, z=Z_surf,
                colorscale='Viridis',
                opacity=0.25,  # Much more transparent for better point visibility
                showscale=False,
                name='Parabolic Dish'
            ))
            
            # Add focus point (smaller for cleaner look)
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[focal_length],
                mode='markers',
                marker=dict(size=8, color='yellow', symbol='diamond'),
                name='Focus Point'
            ))
            
            # Add focal plane (very subtle)
            X_focal, Y_focal, Z_focal = generate_focal_plane(focal_length, dish_radius)
            fig.add_trace(go.Surface(
                x=X_focal, y=Y_focal, z=Z_focal,
                opacity=0.08,  # Very transparent
                colorscale=[[0, 'cyan'], [1, 'cyan']],
                showscale=False,
                name='Focal Plane'
            ))
            
            # Add catacaustic points for current angle (smaller size, enhanced color)
            cat_points = compute_catacaustic_points(angle, focal_length, hole_radius, dish_radius)
            
            if cat_points:
                cat_x = [p[0] for p in cat_points]
                cat_y = [p[1] for p in cat_points]
                cat_z = [p[2] for p in cat_points]
                
                # Get distinct color for this angle with enhanced visibility
                color = get_color_for_angle(angle, min_angle, max_angle)
                
                fig.add_trace(go.Scatter3d(
                    x=cat_x, y=cat_y, z=cat_z,
                    mode='markers',
                    marker=dict(
                        size=0.8,  # Ultra-small points for finest detail
                        color=color,
                        opacity=0.9  # High opacity for good visibility
                    ),
                    name=f'Catacaustic θ={angle:.1f}°'
                ))
            
            # Enhanced scene properties focused on catacaustic region
            scene_dict = dict(
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.8),  # Slightly compressed Z for better viewing
                xaxis=dict(title='X (m)', range=x_range, showgrid=False),
                yaxis=dict(title='Y (m)', range=y_range, showgrid=False),
                zaxis=dict(title='Z (m)', range=z_range, showgrid=False),
                # Enhanced perspective with 45° rotation around z-axis for optimal catacaustic collapse view
                camera=dict(
                    eye=dict(x=3.1, y=0.7, z=1.8),  # Rotated ~45° from (2.2,2.2,1.8)
                    center=dict(x=0, y=0, z=0.3)    # Focus slightly above center
                ),
                bgcolor='rgba(0,0,0,0.9)'
            )
            
            # Update layout with enhanced settings for GIF
            fig.update_layout(
                title=f'Catacaustic Field Collapse - θ={angle:.1f}° (Frame {i+1}/{len(angle_values)})',
                scene=scene_dict,
                height=600,
                width=800,
                template='plotly_dark',
                showlegend=False,  # Clean animation without legend
                margin=dict(l=0, r=0, t=50, b=0)  # Tighter margins
            )
            
            # Convert to image with higher quality settings
            img_bytes = fig.to_image(
                format="png", 
                width=800, 
                height=600, 
                scale=2,  # Higher resolution
                engine="kaleido"
            )
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)
            
            progress_bar.progress((i + 1) / len(angle_values))
        
        # Create enhanced GIF with better settings
        if images:
            gif_buffer = io.BytesIO()
            images[0].save(
                gif_buffer,
                format='GIF',
                save_all=True,
                append_images=images[1:],
                duration=400,  # Faster animation for better flow (400ms per frame)
                loop=0,  # Loop forever
                optimize=True,  # Better compression
                quality=95  # High quality
            )
            gif_buffer.seek(0)
            
            status_text.text("✅ Enhanced catacaustic collapse GIF created!")
            progress_bar.empty()
            
            return gif_buffer.getvalue()
    
    except Exception as e:
        st.error(f"❌ Error creating enhanced GIF: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return None
    
    return None

def main():
    # Initialize session state safely
    if 'animate' not in st.session_state:
        st.session_state.animate = False
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = 0
    if 'angle_values' not in st.session_state:
        st.session_state.angle_values = []
    
    # Header
    st.markdown('<h1 class="main-header">📡 RF Paraboloid Reflector Simulation</h1>', 
                unsafe_allow_html=True)
    
    st.info("✅ Using Plotly for interactive 3D visualization with enhanced catacaustic field rendering")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Reflector Parameters")
    
    # F/D ratio input (primary control)
    f_over_d = st.sidebar.slider(
        "F/D Ratio:",
        min_value=0.1,
        max_value=2.0,
        value=0.25,
        step=0.05,
        help="Focal length to diameter ratio - determines dish depth"
    )
    
    # Dish diameter
    dish_diameter = st.sidebar.number_input(
        "Dish Diameter (m):",
        min_value=0.5,
        max_value=5.0,
        value=1.35,
        step=0.05,
        format="%.2f",
        help="Total diameter of parabolic dish"
    )
    
    # Calculate focal length from F/D ratio
    focal_length = f_over_d * dish_diameter
    st.sidebar.metric("Calculated Focal Length", f"{focal_length:.4f} m")
    
    # Hole radius
    hole_radius = st.sidebar.number_input(
        "Hole Radius (m):",
        min_value=0.0,
        max_value=dish_diameter/4,
        value=0.01524,
        step=0.005,
        format="%.5f",
        help="Radius of central hole in dish"
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
        help="Display focal plane"
    )
    
    show_rotated_plane = st.sidebar.checkbox(
        "Show Rotated Focal Plane",
        value=True,
        help="Display plane perpendicular to incident rays"
    )
    
    # Enhanced Animation controls
    st.sidebar.header("🎬 Enhanced Animation Controls")
    
    animate_angles = st.sidebar.checkbox(
        "Animate Angle Sweep",
        value=False,
        help="Show enhanced catacaustic collapse animation with better viewpoint"
    )
    
    # Multi-angle visualization
    show_multi_angle = st.sidebar.checkbox(
        "Show Multi-Angle View",
        value=False,
        help="Display catacaustic points for multiple angles simultaneously"
    )
    
    # Define angle variables for both modes
    if animate_angles or show_multi_angle:
        angle_start = st.sidebar.slider("Start Angle (deg):", 0.0, 90.0, 50.0, 1.0)
        angle_end = st.sidebar.slider("End Angle (deg):", 0.0, 90.0, 90.0, 1.0)
        angle_steps = st.sidebar.slider("Animation Steps:", 3, 30, 15, 1, 
                                       help="More steps = smoother animation")
        
        # Use current frame angle or default for static display
        if len(st.session_state.angle_values) > 0 and st.session_state.current_frame < len(st.session_state.angle_values):
            incident_angle = st.session_state.angle_values[st.session_state.current_frame]
        else:
            incident_angle = angle_end  # Default to end angle when not animating
        
        if animate_angles:
            col_a, col_b = st.sidebar.columns(2)
            with col_a:
                if st.button("▶️ Play Animation"):
                    st.session_state.animate = True
                    st.session_state.angle_values = linspace(angle_start, angle_end, angle_steps)
                    st.session_state.current_frame = 0
            
            with col_b:
                if st.button("🎬 Enhanced GIF"):
                    st.session_state.create_gif = True
                    st.session_state.gif_angles = linspace(angle_start, angle_end, angle_steps)
                    
            # Enhanced GIF options
            with st.sidebar.expander("🎨 GIF Enhancement Options"):
                st.info("""
                **Enhanced Features:**
                - Smaller point sizes (1.2px) for finer detail
                - Optimized viewing angle for collapse visualization  
                - Static paraboloid with dynamic catacaustic field
                - Focused bounds on catacaustic region
                - Higher quality rendering and compression
                """)
    else:
        # Single angle control - always available
        incident_angle = st.sidebar.slider(
            "Incident Angle (degrees):",
            min_value=0.0,
            max_value=90.0,
            value=90.0,
            step=0.1,
            help="Angle of incoming RF waves"
        )
        # Set default values for animation variables
        angle_start = 50.0
        angle_end = 90.0
        angle_steps = 15
    
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
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.header("📊 Analysis")
        
        # Key metrics
        st.metric("F/D Ratio", f"{f_over_d:.2f}")
        st.metric("Focal Length", f"{focal_length:.4f} m")
        st.metric("Dish Depth", f"{(dish_diameter/2)**2/(4*focal_length):.4f} m")
        st.metric("Dish Area", f"{np.pi * (dish_diameter/2)**2:.3f} m²")
        
        if hole_radius > 0:
            blocked_area = np.pi * hole_radius**2
            efficiency = 1 - (blocked_area / (np.pi * (dish_diameter/2)**2))
            st.metric("Aperture Efficiency", f"{efficiency:.1%}")
        
        # Current angle display - always show the current incident angle
        if not show_multi_angle:
            st.metric("Current Incident Angle", f"{incident_angle:.1f}°")
        else:
            st.metric("Angle Range", f"{angle_start:.1f}° - {angle_end:.1f}°")
        
        # Animation status
        if animate_angles:
            if st.session_state.get('animate', False):
                frame = st.session_state.get('current_frame', 0)
                total_frames = len(st.session_state.get('angle_values', []))
                if total_frames > 0:
                    st.metric("Animation Progress", f"{frame + 1}/{total_frames}")
            else:
                st.info("Click ▶️ to start enhanced animation")
        
        # Single ray analysis
        if analyze_ray and not show_multi_angle:
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
        # Handle enhanced GIF creation
        if st.session_state.get('create_gif', False):
            st.subheader("🎬 Creating Enhanced Catacaustic Collapse GIF")
            
            st.info("""
            **Enhanced GIF Features:**
            🔹 Smaller point sizes for finer catacaustic field detail
            🔹 Optimized viewing angle to best show field collapse
            🔹 Static paraboloid with dynamic point cloud only
            🔹 Focused camera bounds on catacaustic region
            🔹 Higher quality rendering and smoother animation
            """)
            
            gif_data = create_animation_gif(
                st.session_state.gif_angles, 
                focal_length, 
                hole_radius, 
                dish_diameter
            )
            
            if gif_data:
                st.success("✅ Enhanced catacaustic collapse GIF created successfully!")
                
                # Display the GIF
                st.image(gif_data, caption="Enhanced Catacaustic Field Collapse Animation")
                
                # Download button
                st.download_button(
                    label="📥 Download Enhanced GIF",
                    data=gif_data,
                    file_name=f"enhanced_catacaustic_collapse_f{focal_length:.3f}_hole{hole_radius:.3f}.gif",
                    mime="image/gif"
                )
                
                # Show enhanced animation details
                st.info(f"""
                **Enhanced Animation Details:**
                - Angles: {st.session_state.gif_angles[0]:.1f}° → {st.session_state.gif_angles[-1]:.1f}°
                - Frames: {len(st.session_state.gif_angles)}
                - Duration: {len(st.session_state.gif_angles) * 0.4:.1f} seconds
                - Point Size: 0.8px (ultra-fine for maximum detail)
                - View: 45° rotated perspective for optimal catacaustic field collapse
                - Quality: Ultra-high resolution with optimized compression
                """)
            
            st.session_state.create_gif = False
        
        # Handle multi-angle static view
        elif show_multi_angle:
            st.subheader("🌈 Multi-Angle Catacaustic View")
            
            with st.spinner("🔄 Computing enhanced multi-angle visualization..."):
                angle_list = linspace(angle_start, angle_end, angle_steps)
                
                fig = create_multi_angle_visualization(
                    angle_list, focal_length, hole_radius, dish_diameter,
                    show_focal_plane, show_rotated_plane
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"""
            **Enhanced Multi-Angle View:**
            - Showing {len(angle_list)} different incident angles
            - Each angle has a distinct color
            - Point size: 0.8px for enhanced field detail
            - Colors range from blue (low angles) to red (high angles)
            - All catacaustic points displayed simultaneously
            """)
        
        # Handle live animation with enhanced zooming
        elif animate_angles and st.session_state.get('animate', False):
            # Animation loop with enhanced visualization
            frame = st.session_state.get('current_frame', 0)
            
            if frame < len(st.session_state.angle_values):
                current_angle = st.session_state.angle_values[frame]
                
                st.subheader(f"🎬 Enhanced Animation Frame {frame + 1}/{len(st.session_state.angle_values)}")
                st.write(f"**Incident Angle: {current_angle:.1f}°**")
                
                # Generate enhanced visualization for current angle with zoom
                with st.spinner(f"Computing enhanced frame {frame + 1}..."):
                    single_ray_data = None
                    if analyze_ray:
                        single_ray_data = compute_ray_path(current_angle, focal_length, hole_radius, ray_radius, ray_theta)
                    
                    fig = create_plotly_visualization(
                        current_angle, hole_radius, focal_length, dish_diameter,
                        show_incident=False, grid_res=3, show_focal_plane=show_focal_plane, 
                        show_rotated_plane=show_rotated_plane,
                        single_ray_data=single_ray_data, zoom_to_catacaustic=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show enhanced collapse info
                cat_points = compute_catacaustic_points(current_angle, focal_length, hole_radius, dish_diameter/2)
                if cat_points:
                    # Calculate spread of catacaustic points
                    cat_x = [p[0] for p in cat_points]
                    cat_y = [p[1] for p in cat_points]
                    spread = np.sqrt(np.var(cat_x) + np.var(cat_y))
                    st.metric("Catacaustic Spread", f"{spread:.4f} m")
                
                # Auto-advance animation with better timing
                time.sleep(0.6)  # Optimized animation speed
                st.session_state.current_frame = frame + 1
                st.rerun()
            else:
                st.success("✅ Enhanced animation complete!")
                st.balloons()
                st.session_state.animate = False
                st.session_state.current_frame = 0
        
        else:
            # Static visualization with enhanced settings
            with st.spinner("🔄 Computing RF simulation..."):
                single_ray_data = None
                if analyze_ray:
                    single_ray_data = compute_ray_path(incident_angle, focal_length, hole_radius, ray_radius, ray_theta)
                
                fig = create_plotly_visualization(
                    incident_angle, hole_radius, focal_length, dish_diameter,
                    show_incident, grid_res, show_focal_plane, show_rotated_plane,
                    single_ray_data, zoom_to_catacaustic=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.subheader("💾 Export Data")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("📊 Export Catacaustic Points"):
                with st.spinner("Generating catacaustic points..."):
                    if show_multi_angle:
                        # Export all angles
                        all_data = []
                        angle_list = linspace(angle_start, angle_end, angle_steps)
                        for angle in angle_list:
                            cat_points = compute_catacaustic_points(angle, focal_length, hole_radius, dish_diameter/2)
                            for point in cat_points:
                                all_data.append({
                                    'Angle': angle,
                                    'X': point[0],
                                    'Y': point[1],
                                    'Z': point[2]
                                })
                        
                        if all_data:
                            df = pd.DataFrame(all_data)
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Multi-Angle CSV",
                                data=csv,
                                file_name=f"catacaustic_multiangle_{angle_start:.0f}to{angle_end:.0f}deg.csv",
                                mime="text/csv"
                            )
                            st.success(f"✅ Generated {len(all_data)} points for {len(angle_list)} angles")
                    else:
                        # Export single angle
                        cat_points = compute_catacaustic_points(incident_angle, focal_length, hole_radius, dish_diameter/2)
                        if cat_points:
                            df = pd.DataFrame(cat_points, columns=['X', 'Y', 'Z'])
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"catacaustic_points_angle_{incident_angle:.1f}deg.csv",
                                mime="text/csv"
                            )
                            st.success(f"✅ Generated {len(cat_points)} points")
                        else:
                            st.warning("No catacaustic points generated")
        
        with col_export2:
            if animate_angles:
                if st.button("📋 Export Enhanced Animation Data"):
                    all_data = []
                    # Use the same angles as defined in the animation section
                    angles = linspace(angle_start, angle_end, angle_steps)
                    
                    with st.spinner("Computing all enhanced frames..."):
                        for i, angle in enumerate(angles):
                            cat_points = compute_catacaustic_points(angle, focal_length, hole_radius, dish_diameter/2)
                            for point in cat_points:
                                all_data.append({
                                    'Frame': i+1,
                                    'Angle': angle,
                                    'X': point[0],
                                    'Y': point[1],
                                    'Z': point[2]
                                })
                    
                    if all_data:
                        df = pd.DataFrame(all_data)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Enhanced Animation CSV",
                            data=csv,
                            file_name=f"enhanced_catacaustic_animation_{angle_start:.0f}to{angle_end:.0f}deg.csv",
                            mime="text/csv"
                        )
                        st.success(f"✅ Generated {len(all_data)} points across {len(angles)} enhanced frames")

# Run the app
main()
