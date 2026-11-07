"""Locate licensed Times New Roman fonts; never silently substitute another face."""
import os
from pathlib import Path


def times_new_roman_paths():
    roots = [Path(os.environ['TNR_FONT_DIR'])] if os.environ.get('TNR_FONT_DIR') else []
    roots += [Path('/System/Library/Fonts/Supplemental'), Path('/Library/Fonts'),
              Path('C:/Windows/Fonts'), Path.home()/'.local/share/fonts',
              Path('/usr/share/fonts/truetype/msttcorefonts'), Path('/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman')]
    names = {
        'regular': ['Times New Roman.ttf','times.ttf','Times_New_Roman.ttf'],
        'bold': ['Times New Roman Bold.ttf','timesbd.ttf','Times_New_Roman_Bold.ttf'],
        'italic': ['Times New Roman Italic.ttf','timesi.ttf','Times_New_Roman_Italic.ttf'],
        'bolditalic': ['Times New Roman Bold Italic.ttf','timesbi.ttf','Times_New_Roman_Bold_Italic.ttf'],
    }
    found = {}
    for style, candidates in names.items():
        for root in roots:
            match = next((root/name for name in candidates if (root/name).is_file()), None)
            if match:
                found[style] = match
                break
    if len(found) != 4:
        raise FileNotFoundError('Install licensed Times New Roman regular, bold, italic and bold-italic TTF files, then set TNR_FONT_DIR to their directory. Font files are not redistributed with this repository.')
    return found


def register_report_fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    names = {'regular':'TNR','bold':'TNR-Bold','italic':'TNR-Italic','bolditalic':'TNR-BoldItalic'}
    for style,path in times_new_roman_paths().items():
        pdfmetrics.registerFont(TTFont(names[style],str(path)))
    pdfmetrics.registerFontFamily('TNR',normal='TNR',bold='TNR-Bold',italic='TNR-Italic',boldItalic='TNR-BoldItalic')


def configure_plot_fonts():
    import matplotlib as mpl
    from matplotlib import font_manager
    for path in times_new_roman_paths().values():
        font_manager.fontManager.addfont(str(path))
    mpl.rcParams.update({'font.family':'Times New Roman','font.serif':['Times New Roman'],
                         'mathtext.fontset':'custom','mathtext.rm':'Times New Roman',
                         'mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold',
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False})
