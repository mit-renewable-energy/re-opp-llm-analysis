"""
Color style guide for publication-ready plots
Contains the specified color palette for paper figures
"""

import matplotlib.pyplot as plt

# Publication color palette
class PaperColors:
    """Color palette for publication-ready figures"""
    
    # Main color palette
    CORNFLOWER = "#8ECBE6"
    EASTERN_BLUE = "#229EBC"
    GREEN_VOGUE = "#023047" 
    BLUMINE = "#1E5571"
    SELECTIVE_YELLOW = "#FFB706"
    FLUSH_ORANGE = "#FC8500"
    CORNFLOWER_BLUE = "#5E8CF1"
    WHITE = "#FFFFFF"
    SILVER = "#C0C0C0"
    GALLERY = "#EBEBEB"
    THUNDERBIRD = "#C1131F"
    RED_BERRY = "#980100"
    SERENADE = "#FFF4E5"
    
    # Technology-specific colors
    SOLAR_COLOR = FLUSH_ORANGE  # Orange for solar
    WIND_COLOR = BLUMINE        # Dark blue for wind (to differentiate from support blue)
    
    # Opposition/Support colors  
    OPPOSITION_COLOR = THUNDERBIRD  # Red for opposition
    SUPPORT_COLOR = EASTERN_BLUE    # Blue for support
    NEUTRAL_COLOR = SILVER          # Gray as needed
    
    # Color lists for multi-category plots
    PRIMARY_PALETTE = [
        CORNFLOWER, EASTERN_BLUE, GREEN_VOGUE, BLUMINE,
        SELECTIVE_YELLOW, FLUSH_ORANGE, CORNFLOWER_BLUE
    ]
    
    EXTENDED_PALETTE = [
        CORNFLOWER, EASTERN_BLUE, GREEN_VOGUE, BLUMINE,
        SELECTIVE_YELLOW, FLUSH_ORANGE, CORNFLOWER_BLUE,
        SILVER, THUNDERBIRD, RED_BERRY
    ]

# Publication formatting settings
class PaperFormat:
    """Publication formatting specifications"""
    
    # Font settings for A4 portrait page
    FONT_FAMILY = "Arial"
    FONT_SIZE_SMALL = 8      # Base text size for A4
    FONT_SIZE_MEDIUM = 9     # Axis labels
    FONT_SIZE_LARGE = 10     # Titles
    
    # Figure dimensions for A4 portrait (inches)
    # A4 width ≈ 8.27 inches, with margins leaves ~7 inches usable
    # A4 height ≈ 11.7 inches, half-page ≈ 5 inches usable
    FIG_WIDTH_SINGLE = 3.5   # Half-width figure
    FIG_WIDTH_DOUBLE = 7.0   # Full-width figure
    FIG_HEIGHT_STANDARD = 4.0 # Half-page height
    FIG_HEIGHT_TALL = 5.0    # Taller plots
    
    # Output settings
    DPI = 300
    FORMAT = 'png'
    
    # Plot styling
    LINE_WIDTH = 1.0
    MARKER_SIZE = 3
    ALPHA_OVERLAY = 0.7

def setup_paper_style():
    """Set up matplotlib for publication-ready plots"""
    plt.style.use('default')  # Reset to default first
    
    # Configure matplotlib
    plt.rcParams.update({
        # Font settings
        'font.family': PaperFormat.FONT_FAMILY,
        'font.size': PaperFormat.FONT_SIZE_SMALL,
        'axes.labelsize': PaperFormat.FONT_SIZE_MEDIUM,
        'axes.titlesize': PaperFormat.FONT_SIZE_LARGE,
        'xtick.labelsize': PaperFormat.FONT_SIZE_SMALL,
        'ytick.labelsize': PaperFormat.FONT_SIZE_SMALL,
        'legend.fontsize': PaperFormat.FONT_SIZE_SMALL,
        
        # Figure settings
        'figure.dpi': PaperFormat.DPI,
        'savefig.dpi': PaperFormat.DPI,
        'savefig.format': PaperFormat.FORMAT,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        
        # Plot appearance
        'axes.linewidth': 0.5,
        'axes.edgecolor': 'black',
        'axes.facecolor': 'white',
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # Lines and markers
        'lines.linewidth': PaperFormat.LINE_WIDTH,
        'lines.markersize': PaperFormat.MARKER_SIZE,
        'patch.linewidth': 0.5,
        
        # Color cycle
        'axes.prop_cycle': plt.cycler(color=PaperColors.PRIMARY_PALETTE)
    })

def save_paper_figure(filename, folder_path, fig=None, **kwargs):
    """Save figure in publication format to specified folder"""
    import os
    
    # Default save parameters
    save_params = {
        'dpi': PaperFormat.DPI,
        'format': PaperFormat.FORMAT,
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    save_params.update(kwargs)
    
    # Ensure folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Full path
    full_path = os.path.join(folder_path, filename)
    
    # Save the figure
    if fig is None:
        plt.savefig(full_path, **save_params)
    else:
        fig.savefig(full_path, **save_params)
    
    print(f"📁 Saved publication figure: {full_path}")

def create_color_reference():
    """Create a visual reference of the color palette"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(PaperFormat.FIG_WIDTH_DOUBLE, 6))
    
    # Primary palette
    colors1 = PaperColors.PRIMARY_PALETTE
    names1 = ['Cornflower', 'Eastern Blue', 'Green Vogue', 'Blumine', 
              'Selective Yellow', 'Flush Orange', 'Cornflower Blue']
    
    for i, (color, name) in enumerate(zip(colors1, names1)):
        ax1.barh(i, 1, color=color, edgecolor='black', linewidth=0.5)
        ax1.text(0.5, i, f'{name}\n{color}', ha='center', va='center', 
                fontsize=PaperFormat.FONT_SIZE_SMALL, weight='bold')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.5, len(colors1) - 0.5)
    ax1.set_title('Primary Color Palette', fontsize=PaperFormat.FONT_SIZE_LARGE)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # Special purpose colors
    special_colors = [PaperColors.SOLAR_COLOR, PaperColors.WIND_COLOR, 
                     PaperColors.OPPOSITION_COLOR, PaperColors.SUPPORT_COLOR,
                     PaperColors.NEUTRAL_COLOR]
    special_names = ['Solar (Orange)', 'Wind (Blue)', 'Opposition (Red)', 
                    'Support (Blue)', 'Neutral (Gray)']
    
    for i, (color, name) in enumerate(zip(special_colors, special_names)):
        ax2.barh(i, 1, color=color, edgecolor='black', linewidth=0.5)
        ax2.text(0.5, i, f'{name}\n{color}', ha='center', va='center',
                fontsize=PaperFormat.FONT_SIZE_SMALL, weight='bold', 
                color='white' if color in [PaperColors.GREEN_VOGUE, PaperColors.RED_BERRY] else 'black')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.5, len(special_colors) - 0.5)
    ax2.set_title('Special Purpose Colors', fontsize=PaperFormat.FONT_SIZE_LARGE)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Demo the color palette
    setup_paper_style()
    fig = create_color_reference()
    plt.show()