import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Advanced Equation Solver - Topic G",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
    }
    .root-card {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

def bisection_method(f, a, b, tol=1e-8, max_iter=200):
    """
    Enhanced bisection method with comprehensive error handling.
    """
    # Input validation
    if a >= b:
        return None, 0, "Error: Lower bound 'a' must be less than upper bound 'b'."
    
    if tol <= 0:
        return None, 0, "Error: Tolerance must be positive."
    
    fa, fb = f(a), f(b)
    
    # Check for root at endpoints
    if abs(fa) < tol:
        return a, 0, "Root found at lower bound."
    if abs(fb) < tol:
        return b, 0, "Root found at upper bound."
    
    # Check sign change
    if fa * fb > 0:
        return None, 0, "No root found: function has same sign at endpoints."
    
    # Bisection algorithm
    iter_count = 0
    for iter_count in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        
        # Check convergence
        if abs(fc) < tol:
            return c, iter_count, f"Converged after {iter_count} iterations"
        
        # Update interval
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    
    return c, max_iter, f"Maximum iterations reached"

def find_all_roots(f, x_min, x_max, step=0.01, tol=1e-8, max_iter_per_root=200):
    """
    Find all roots in a given range.
    """
    roots = []
    status_messages = []
    
    # Scan for potential root intervals
    x_current = x_min
    while x_current < x_max:
        x_next = min(x_current + step, x_max)
        
        fa, fb = f(x_current), f(x_next)
        
        # Check for sign change
        if fa * fb <= 0:
            root, iters, status = bisection_method(f, x_current, x_next, tol, max_iter_per_root)
            
            if root is not None:
                # Check for duplicates
                is_duplicate = False
                for existing_root in roots:
                    if abs(root - existing_root) < tol:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    roots.append(root)
                    status_messages.append(f"Root {len(roots)}: x = {root:.8f} | {status}")
        
        x_current = x_next
    
    if roots:
        status_messages.insert(0, f"Found {len(roots)} root(s) in range [{x_min}, {x_max}]")
    else:
        status_messages.insert(0, "No roots found. Try adjusting search range.")
    
    return roots, status_messages

def equation_to_solve(x, a, b, c, w, v):
    """Define the equation: a*x^b = e^(c*x) * sin(w*x + v)"""
    left_side = a * (x ** b)
    right_side = np.exp(c * x) * np.sin(w * x + v)
    return left_side - right_side

def main():
    # Header
    st.markdown('<div class="main-header">🎯 Advanced Equation Solver - Topic G</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Solve equations of the form:**  
    `a × xᵇ = e^(c·x) × sin(w·x + v)`  
    *Using the robust Successive Bisection Method*
    """)
    
    # Sidebar - User Inputs
    st.sidebar.header("🔧 Equation Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.number_input("Constant **a**", value=1.0, step=0.1, help="Multiplicative constant")
        b = st.number_input("Power **b**", value=1.0, step=0.1, help="Exponent of x")
        c = st.number_input("Exponent **c**", value=0.1, step=0.1, help="Coefficient in the exponential term")
    
    with col2:
        w = st.number_input("Frequency **w**", value=5.0, step=0.1, help="Frequency of the sine function")
        v = st.number_input("Phase **v**", value=0.0, step=0.1, help="Phase shift of the sine function")
    
    st.sidebar.header("🎯 Solver Settings")
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        x_min = st.number_input("Min x", value=-5.0, step=0.5)
        x_max = st.number_input("Max x", value=5.0, step=0.5)
    
    with col4:
        step_size = st.number_input("Step Size", value=0.01, min_value=0.001, max_value=1.0, step=0.001, 
                                  help="Smaller values find roots more reliably but take longer")
        tolerance = st.number_input("Tolerance", value=1e-8, format="%e", 
                                  help="Desired precision for root finding")
    
    max_iter = st.sidebar.slider("Max Iterations", 50, 1000, 200, 
                               help="Maximum number of bisection iterations for each root")
    
    # Main content area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Solve button
        if st.button("🔍 Find All Roots!", type="primary", use_container_width=True):
            with st.spinner('Scanning for roots... This may take a moment.'):
                # Define the function with current parameters
                def current_equation(x):
                    return equation_to_solve(x, a, b, c, w, v)
                
                # Find all roots
                roots, status_messages = find_all_roots(current_equation, x_min, x_max, step_size, tolerance, max_iter)
                
                # Display results
                st.header("📊 Results")
                
                if roots:
                    # Success message
                    st.success(f"**Found {len(roots)} unique root(s)!**")
                    
                    # Display roots in cards
                    for i, root in enumerate(roots):
                        with st.container():
                            st.markdown(f"""
                            <div class="root-card">
                                <h4>Root {i+1}</h4>
                                <h3>x = {root:.8f}</h3>
                                <p>Verification: f(x) = {current_equation(root):.2e}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'Root Number': range(1, len(roots) + 1),
                        'x_value': roots,
                        'f(x)': [current_equation(root) for root in roots]
                    })
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="roots_results.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.error("No roots found in the specified range.")
                
                # Detailed solver log
                with st.expander("📋 Detailed Solver Log"):
                    for message in status_messages:
                        st.write(message)
                
                # Visualization
                st.header("📈 Function Visualization")
                
                # Generate plot data
                x_plot = np.linspace(x_min, x_max, 1000)
                y_plot = current_equation(x_plot)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot function
                ax.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'f(x) = {a}·x^{b} - e^({c}·x)·sin({w}·x + {v})', alpha=0.8)
                ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='y = 0')
                
                # Plot roots
                if roots:
                    roots_y = current_equation(np.array(roots))
                    ax.plot(roots, roots_y, 'ro', markersize=8, label=f'Found Roots ({len(roots)})', zorder=5)
                
                # Plot styling
                ax.set_xlabel('x')
                ax.set_ylabel('f(x)')
                ax.set_title('Root Finding using Successive Bisection Method')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
                
                # Download plot
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                st.download_button(
                    label="📥 Download Plot",
                    data=buf.getvalue(),
                    file_name="function_plot.png",
                    mime="image/png"
                )
    
    with col_right:
        st.header("💡 Quick Guide")
        
        st.markdown("""
        <div class="info-box">
        <h4>🎯 How to Use:</h4>
        <ol>
        <li>Set equation parameters</li>
        <li>Define search range</li>
        <li>Click "Find All Roots!"</li>
        <li>Analyze results & plot</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.header("📚 Examples")
        
        if st.button("Simple Example", use_container_width=True):
            st.info("Try: a=1, b=1, c=0, w=1, v=0")
        
        if st.button("Complex Example", use_container_width=True):
            st.info("Try: a=1, b=2, c=-0.2, w=5, v=1")
        
        st.header("🔍 Tips")
        st.markdown("""
        - **Multiple Roots**: Use smaller step sizes
        - **No Roots**: Increase search range
        - **Precision**: Lower tolerance values
        """)

if __name__ == "__main__":
    main()