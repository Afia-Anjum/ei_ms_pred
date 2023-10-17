import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from pyrsistent import b
from rdkit.Chem import Draw, AllChem
from copy import deepcopy

from cfmtoolkit.datastructures import Spectrum
from cfmtoolkit.datastructures.cfm_spectra import CfmSpectra
from cfmtoolkit.metrics import Comparators
from cfmtoolkit.utils import Utilities
from cfmtoolkit.standardization import Standardization
from vis_utils import *
from math import ceil
from refined_formula_guessor import guess_formula

import pickle
# FuncFormatter can be used as a decorator
@mticker.FuncFormatter
def spectra_major_formatter(x, pos):
    x = -x if x < 0 else x
    return "%d" % x + '%'


def correct_the_intensity(file_name,length):
    with open(file_name) as f:
        content_list4 = f.readlines()


    content_list4 = [(x.strip()) for x in content_list4]
    m_by_z = []
    intensity = []
    for i in range(len(content_list4)):
        p=content_list4[i].split()
        m_by_z.append(float(p[0]))
        if "orig" in file_name:
            intensity.append(float(p[1]))
        else:
            intensity.append(float(p[1]) * 10)
    max_m_by_z = length
    max_intensity = max(intensity)

    correct_m_by_z = []
    correct_intensity = []

    for i in range(0, int(max_m_by_z) + 1):
        correct_m_by_z.append(i)
        correct_intensity.append(float(0))


    for i in range(len(intensity)):
        correct_intensity[int(m_by_z[i])] = intensity[i]

    return correct_intensity

def correct_the_intensity2(file_name,length):
    # read the spectra and convert them into a continuous list of m/z and intensities
    with open(file_name) as f:
        content_list4 = f.readlines()
    content_list4 = [x.strip() for x in content_list4]


    m_by_z = []
    intensity = []
    for i in range(len(content_list4)):
        p = content_list4[i].split()
        m_by_z.append(float(p[0]))

        intensity.append(float(p[1]))

    max_intensity = max(intensity)
    for i in range(len(intensity)):
        intensity[i]=(intensity[i]/max_intensity)*1000

    max_m_by_z = length

    correct_m_by_z = []
    correct_intensity = []

    for i in range(0, int(max_m_by_z) + 1):
        correct_m_by_z.append(i)
        correct_intensity.append(float(0))


    for i in range(len(intensity)):
        correct_intensity[int(m_by_z[i])] = intensity[i]
    return correct_intensity

class SpectraVis(object):
    _annotation_top_kws = {
        'horizontalalignment': 'right',
        'verticalalignment': 'center', 'rotation': 90,
        'rotation_mode': 'anchor'}

    _annotation_bottom_kws = {
        'horizontalalignment': 'left',
        'verticalalignment': 'center', 'rotation': 90,
        'rotation_mode': 'anchor'}

    _default_height = 130
    _default_summary_font_size = 20
    _default_info_block_font_size = 24
    _default_info_block_title_font_size = 36
    _default_title_font_size = 24
    _default_tick_font_size = 16
    _default_annotation_font_size = 10
    _default_legends_font_size = 20
    _default_info_block_margin = 600
    _default_mol_image_size = (600, 300)
    _default_ax_width = 12
    _default_ax_height = 3
    _default_info_block_height = 1
    _default_peak_width = 2.0
    _default_font_family = 'sans-serif'
    _default_fonts = ['Arial']
    _default_bar_alpha = 1.0

    _colors = {'default': '#212121', 'text': '#212121', 'metric_background': '#0072B2'}

    _peak_colors = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377', '#BBBBBB']
    _baseline_colors = ["#56B4E970", '#009E7370']
    _cfm_colors = ['#1d1aeb', '#E63434']
    _indicators = {'up': '▲', 'down': '▼'}
    _default_dpi = 128

    @classmethod
    def get_mol_img_from_chemid(cls, chem_id_str: str, size=None, options=None,
                                mol_template_coords: "Compute2DCoords" = None):
        '''
        Method draw molecule by Smiles or Inchi
        retuin an PIL image
        '''

        mol = Utilities.create_mol_by_chemid(chem_id_str)
        if size is None:
            size = cls._default_mol_image_size

        # align mol with template coords
        if mol_template_coords is not None:
            try:
                AllChem.GenerateDepictionMatching2DStructure(mol, mol_template_coords)
            except ValueError:
                pass

        # print(size)
        if options == None:
            options = Draw.MolDrawOptions()
            options.bondLineWidth = 6
            options.minFontSize = 36
            options.maxFontSize = 128
            options.noCarbonSymbols = True
            options.backgroundColour = None
        mol_drawing = Draw.MolToImage(mol, size=size, fitImage=False, options=options)
        return mol_drawing

    @classmethod
    def plot_mol_from_smiles_or_inchi(cls, chem_id_str: str):
        '''
        Method draw molecule by Smiles or Inchi
        retuin an PIL image
        '''
        fig = plt.figure(figsize=(cls._default_ax_width, cls._default_ax_height))
        ax = fig.gca()
        cls._plot_info_block_axes(ax, chem_id_str)
        return fig

    '''
    Method to plot a spectrum
    '''

    @classmethod
    def plot_mirrored_spectrum(cls, top_spectrum: Spectrum, bottom_spectrum: Spectrum, text_type: str):
        fig = plt.figure(figsize=(cls._default_ax_width, cls._default_ax_height))
        ax = fig.gca()

        cls._plot_spectrum_axes(
            ax, top_spectrum, bottom_spectrum, None, spectrum_plot_settings={'text_type': text_type})
        return fig

    '''
    Method to draw peaks
    very low peak will be focus to draw with 1.0% height
    '''

    @classmethod
    def _draw_peaks(cls, ax, peaks, max_intensity, label,
                    peak_width,
                    inverted=False,
                    peak_color=None,
                    show_annotation=False):

        for idx, peak in enumerate(peaks):
            mz = peak.mz
            # doing so very low peak will show up
            display_peak_intensity = max(1.0, peak.intensity / max_intensity * 100.0)
            annotation_pos = display_peak_intensity + 5
            if inverted:
                display_peak_intensity *= -1
                annotation_pos *= -1

            if peak_color is None:
                if inverted:
                    peak_color = cls._peak_colors[1]
                else:
                    peak_color = cls._peak_colors[0]

            # Draw peak
            if idx == 0:
                ax.bar(mz, display_peak_intensity, alpha=cls._default_bar_alpha, width=peak_width, color=peak_color,
                       label=label)
            else:
                ax.bar(mz, display_peak_intensity, alpha=cls._default_bar_alpha, width=peak_width, color=peak_color)

            # Draw Annotation as Text

            if peak.annotation != Standardization.NA and show_annotation:
                annotation_str = str(peak.annotation)
                ax.text(mz, annotation_pos, annotation_str,
                        fontsize=cls._default_legends_font_size, color=peak_color,
                        horizontalalignment='center', verticalalignment='center', weight='bold')

    '''
    Method draw a spectrum on matplot figure axis
    '''

    @classmethod
    def _plot_spectrum_axes(cls, ax, top_spectrum: Spectrum,
                            bottom_spectrum: Optional[Spectrum] = None,
                            metrics: Optional[dict] = None,
                            spectrum_plot_settings: Optional[dict] = None, ):

        if spectrum_plot_settings == None:
            spectrum_plot_settings = {}
        text_type = spectrum_plot_settings.get('text_type')
        top_peak_color = spectrum_plot_settings.get('top_peak_color')
        bottom_peak_color = spectrum_plot_settings.get('bottom_peak_color')
        min_mz = spectrum_plot_settings.get('min_mz')
        max_mz = spectrum_plot_settings.get('max_mz')

        show_x_label = spectrum_plot_settings.get('show_x_label', True)
        show_y_label = spectrum_plot_settings.get('show_y_label', True)
        show_legend = spectrum_plot_settings.get('show_legend', True)
        show_annotation = spectrum_plot_settings.get('show_annotation', False)
        summary_font_size = spectrum_plot_settings.get('summary_font_size', cls._default_summary_font_size)

        if show_x_label:
            ax.set_xlabel('m/z', fontsize=cls._default_annotation_font_size)
        if show_y_label:
            ax.set_ylabel('Relative Intensity', fontsize=cls._default_annotation_font_size)

        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')

        # set ranges
        mz_end = top_spectrum.max_mz + 50

        y_min = 0
        y_max = 100 if not show_annotation else cls._default_height

        if bottom_spectrum is not None:
            mz_end = max(mz_end, bottom_spectrum.max_mz + 50)
            y_min = -100 if not show_annotation else -cls._default_height
            ax.xaxis.set_label_coords(1.05, -y_min / (y_max - y_min))

        if max_mz is not None:
            mz_end = max_mz
        if min_mz is not None:
            mz_start = min_mz
        else:
            mz_start = 1

        peak_width = 0.5 * (max_mz / 300)  # cls._default_peak_width

        cls._draw_peaks(ax, top_spectrum.peaks,
                        top_spectrum.max_intensity, top_spectrum.get_spectrum_type_summary(), peak_width,
                        inverted=False,
                        peak_color=top_peak_color, show_annotation=show_annotation)

        if bottom_spectrum is not None:
            cls._draw_peaks(ax, bottom_spectrum.peaks,
                            bottom_spectrum.max_intensity, bottom_spectrum.get_spectrum_type_summary(), peak_width,
                            inverted=True, peak_color=bottom_peak_color, show_annotation=show_annotation)

        # print(mz_end, max_mz)
        ax.set_xlim(left=mz_start, right=mz_end)
        ax.set_ylim(ymin=y_min, ymax=y_max)

        ax.yaxis.set_major_formatter(spectra_major_formatter)

        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='both', labelsize='small')

        # add scores
        if metrics is not None:
            metrics_text = ""
            for idx, label in enumerate(metrics):
                if idx > 0:
                    metrics_text += '\n'

                if type(metrics[label]) is float:
                    metrics_text += "{}:{:.2}".format(label, metrics[label])
                elif type(metrics[label]) is int:
                    metrics_text += "{}:{}".format(label, metrics[label])
                elif type(metrics[label]) is str:
                    metrics_text += "{}:{}".format(label, metrics[label])

            ax.text(10, y_max - 5, metrics_text,
                    horizontalalignment='left',
                    verticalalignment='top',
                    # transform=ax.transAxes,
                    fontsize=summary_font_size,
                    linespacing=1.5,
                    # family= 'monospace',
                    # color=color_predicted,
                    bbox=dict(facecolor='0.75',  # bottom_peak_color,  # cls._colors['metric_background'],
                              # edgecolor= bottom_peak_color,
                              alpha=0.3, boxstyle='round,pad=0.3'))

        # print(text_type)
        if text_type is not None:
            # font_size = cls._default_summary_font_size

            top_spectrum_text = cls._get_spectrum_display_text(
                top_spectrum, text_type)
            ax.text(3, cls._default_height - summary_font_size, top_spectrum_text,
                    color=cls._colors['default'], fontsize=summary_font_size)

        if bottom_spectrum is not None and text_type is not None:
            bottom_spectrum_text = cls._get_spectrum_display_text(
                bottom_spectrum, text_type)

            ax.text(3, -(cls._default_height - summary_font_size), bottom_spectrum_text,
                    color=cls._colors['default'], fontsize=summary_font_size)

        y_ticks = ax.get_yticks()
        ax.set_yticks(y_ticks[(-100 <= y_ticks) & (100 >= y_ticks)])
        ax.tick_params(axis='both', which='major', labelsize=cls._default_tick_font_size)
        # bbox_to_anchor=(1.00, 1.1)
        if show_legend:
            ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.2), fontsize=cls._default_legends_font_size,
                      shadow=False,
                      ncol=1, frameon=False)

        #print("FINISH!")
    @classmethod
    def _get_spectrum_display_text(cls, spectrum: Spectrum, text_type: str):
        text = ""
        if spectrum is None:
            return ""
        # if text_type == "spectrum_type_summary":
        #    text = spectrum.get_spectrum_type_summary()
        if text_type == "summary":
            text = spectrum.get_summary()
        return text

    @staticmethod
    def _get_compound_info_text(compound_name: str = '<n/a>', inchikey: str = '<n/a>'):
        return "Name: {} \nInchIKey: {} \n".format(compound_name, inchikey)

    @classmethod
    def _plot_info_block_axes(cls, ax,
                              smiles_or_inchi: str,
                              info_block_title: str = None,
                              info_text: str = None,
                              metrics: Dict[str, float] = None,
                              info_block_title_font_size: int = None,
                              info_block_font_size: int = None,
                              mol_image_options=None,
                              mol_template_coords=None,
                              image_width_resize_factor=None):

        if info_block_title_font_size is None:
            info_block_title_font_size = cls._default_info_block_title_font_size
        if info_block_font_size is None:
            info_block_font_size = cls._default_info_block_font_size

        fig = plt.gcf()

        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax_width, ax_height = bbox.width, bbox.height
        ax_width *= fig.dpi
        ax_height *= fig.dpi

        ax.set_xlim(0, ax_width)
        ax.set_ylim(ax_height, 0)

        y_offset = 16
        # Title
        if info_block_title is not None:
            text = ax.text(0, y_offset, info_block_title,
                           horizontalalignment='left',
                           verticalalignment='top',
                           color=cls._colors['default'],
                           fontsize=info_block_title_font_size)
            text_bbox = text.get_window_extent(find_renderer(fig))
            y_offset += text_bbox.height * 1.1
        # Details
        if info_text is not None:
            text = ax.text(0, y_offset, info_text,
                           horizontalalignment='left',
                           verticalalignment='top',
                           color=cls._colors['default'],
                           fontsize=info_block_font_size)
            text_bbox = text.get_window_extent(find_renderer(fig))
            y_offset += text_bbox.height * 1.5

        # show mertics
        mertics_bar_height = info_block_font_size * 1.5
        metrics_block_height = 0
        if metrics is not None:
            mertics_idx = 0
            for metric_label in metrics:
                metric_bar_loc = mertics_idx * mertics_bar_height + y_offset
                ax.barh(metric_bar_loc, metrics[metric_label] * ax_width, mertics_bar_height, alpha=0.5)
                ax.text(0, metric_bar_loc, "{}:{:.2f}".format(metric_label, metrics[metric_label]),
                        horizontalalignment='left',
                        verticalalignment='center', fontsize=info_block_font_size)

                mertics_idx += 1
                metrics_block_height += mertics_bar_height

        if info_block_title is None and info_text is None and metrics_block_height == 0:
            y_offset = 0

        if image_width_resize_factor is None:
            image_width_resize_factor = 1.0

        image_width = int(ax_width / image_width_resize_factor) * 2
        image_height = int(ax_height - metrics_block_height - y_offset) * 2
        # print("image_width: {}".format(image_width))
        # print("image_height: {}".format(image_height))
        # print(ax_height, metrics_block_height, y_offset)

        mol_image = cls.get_mol_img_from_chemid(smiles_or_inchi,
                                                size=(image_width, image_height),
                                                options=mol_image_options,
                                                mol_template_coords=mol_template_coords)
        # extent = (left, right, bottom, top)
        extent = (0, image_width / 2, ax_height, ax_height - image_height / 2)
        ax.imshow(mol_image, extent=extent, interpolation='lanczos')
        ax.axis('off')

    @classmethod
    def _init_figure_and_info_block(cls, num_row, num_col, smiles_or_inchi, info_text,
                                    width_scale=1,
                                    height_scale=1,
                                    height_ratios=None):
        # print(height_ratios)
        if smiles_or_inchi is not None:
            num_row += 1

        if height_ratios is None:
            height_ratios = [2] * num_row
            if smiles_or_inchi is not None:
                height_ratios[0] = 2

        # draw figures per ev level
        fig = plt.figure(figsize=(num_col * cls._default_ax_width * width_scale,
                                  num_row * cls._default_ax_height * height_scale), dpi=cls._default_dpi)
        # print(num_col *  cls._default_ax_width * width_scale,
        #                          num_row * cls._default_ax_height * height_scale)
        # print('_init_figure_and_info_block',num_row, num_col)
        # print(num_col, num_row, height_ratios)
        gs = GridSpec(ncols=num_col, nrows=num_row, figure=fig, height_ratios=height_ratios)

        # draw mol
        cur_row = 0
        if smiles_or_inchi is not None:
            ax = fig.add_subplot(gs[cur_row, 0:])
            cls._plot_info_block_axes(ax, smiles_or_inchi, info_text=info_text)
            cur_row += 1

        return fig, gs, cur_row

    @classmethod
    def plot_spectra(cls, spectra: List[Spectrum], smiles_or_inchi=None, info_text=None, show_annotation=False):
        """Method to plot a list of spectra

        Args:
            spectra (List[Spectrum]): [description]
            smiles_or_inchi ([type], optional): [description]. Defaults to None.
            info_text ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """

        num_row = len(spectra)
        num_col = 1
        fig, gs, cur_row = cls._init_figure_and_info_block(num_row, num_col, smiles_or_inchi, info_text)

        if smiles_or_inchi is not None:
            num_row += 1

        max_mz = 0
        for spectrum in spectra:
            max_mz = max(spectrum.max_mz, max_mz)
        # add right margin
        max_mz += 50

        for spectrum in spectra:
            ax = fig.add_subplot(gs[cur_row, 0])
            spectrum_plot_settings = {
                'text_type': "spectrum_type_summary",
                'max_mz': max_mz,
                'show_annotation': show_annotation
            }
            cls._plot_spectrum_axes(ax, spectrum, spectrum_plot_settings=spectrum_plot_settings)
            cur_row += 1

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)
        return fig

    @classmethod
    def plot_cfm_spectra(cls, cfm_spectra: CfmSpectra, show_annotation: bool = False):
        """Method to plot a cfm spectra

        Args:
            cfm_spectra (CfmSpectra): [description]

        Returns:
            [type]: [description]
        """

        num_row = 3
        num_col = 1
        smiles_or_inchi = cfm_spectra.smiles_or_inchi
        info_text = cfm_spectra.inchikey
        # print(smiles_or_inchi)
        fig, gs, cur_row = cls._init_figure_and_info_block(num_row, num_col, smiles_or_inchi, info_text, height_scale=2)

        if smiles_or_inchi is not None:
            num_row += 1

        max_mz = 0
        for spectrum in cfm_spectra.get_spectra_as_list():
            max_mz = max(spectrum.max_mz, max_mz)
        # add right margin
        max_mz += 50

        for spectrum in cfm_spectra.get_spectra_as_list():
            ax = fig.add_subplot(gs[cur_row, 0])

            spectrum_plot_settings = {
                'text_type': "spectrum_type_summary",
                'max_mz': max_mz,
                'show_annotation': show_annotation
            }
            cls._plot_spectrum_axes(ax, spectrum, spectrum_plot_settings=spectrum_plot_settings)

            cur_row += 1

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)
        return fig

    @classmethod
    def plot_mirrored_spectra(cls, groundtruth_spectra: List[Spectrum],
                              predicted_spectra_list: List[List[Spectrum]],
                              predicted_spectra_smiles_or_inchi_list: List[str] = None,
                              predicted_spectra_info_text: List[str] = None,
                              spectra_metrics_list: List[List[Dict]] = None,
                              smiles_or_inchi: str = None,
                              info_text: str = None,
                              show_mol_image_for_each_col=False,
                              show_annotation: bool = False):

        """ Method to create figure to compare many predicted spectra vs the same ground truth side by side

        Args:
            groundtruth_spectra (List[Spectrum]): [description]
            predicted_spectra_list (List[List[Spectrum]]): [description]
            predicted_spectra_smiles_or_inchi_list (List[str], optional): [description]. Defaults to None.
            predicted_spectra_info_text (List[str], optional): [description]. Defaults to None.
            spectra_metrics_list (List[List[Dict]], optional): [description]. Defaults to None.
            smiles_or_inchi (str, optional): [description]. Defaults to None.
            info_text (str, optional): [description]. Defaults to None.
            show_mol_image_for_each_col (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """

        num_row = len(groundtruth_spectra)
        num_col = len(predicted_spectra_list)
        # height_ratios = [2] * num_row
        max_mz = 0
        for spectra in predicted_spectra_list + [groundtruth_spectra]:
            for spectrum in spectra:
                # print(spectrum)
                # print(spectrum.max_mz)
                max_mz = max(spectrum.max_mz, max_mz)

        max_mz += 50

        # init fig
        has_predicted_smiles = predicted_spectra_smiles_or_inchi_list is not None
        plot_predicted_mol_block = show_mol_image_for_each_col and has_predicted_smiles
        # print(num_row, num_col)
        if plot_predicted_mol_block:
            num_row += 1

        has_metrics = spectra_metrics_list is not None
        fig, gs, cur_row = cls._init_figure_and_info_block(num_row, num_col, smiles_or_inchi, info_text,
                                                           height_scale=2.0)
        display_mol_template = None
        if smiles_or_inchi is not None and plot_predicted_mol_block:
            # create a 2d coords template
            display_mol_template = Utilities.create_mol_by_chemid(smiles_or_inchi)
            AllChem.Compute2DCoords(display_mol_template)
        if smiles_or_inchi is None and plot_predicted_mol_block:
            # create a 2d coords template
            display_mol_template = Utilities.create_mol_by_chemid(predicted_spectra_smiles_or_inchi_list[0])
            AllChem.Compute2DCoords(display_mol_template)

        for col_idx, spectra in enumerate(predicted_spectra_list):
            if plot_predicted_mol_block:
                ax = fig.add_subplot(gs[cur_row, col_idx])
                text_info = predicted_spectra_info_text[col_idx] if predicted_spectra_info_text is not None else ''
                cls._plot_info_block_axes(ax, predicted_spectra_smiles_or_inchi_list[col_idx], info_text=text_info,
                                          info_block_font_size=cls._default_info_block_font_size,
                                          mol_template_coords=display_mol_template,
                                          )

            for gt_spectrum_idx in range(len(groundtruth_spectra)):
                row_idx = gt_spectrum_idx + cur_row
                if show_mol_image_for_each_col:
                    row_idx += 1
                ax = fig.add_subplot(gs[row_idx, col_idx])

                top_spectrum = groundtruth_spectra[gt_spectrum_idx]
                bottom_spectrum = predicted_spectra_list[col_idx][gt_spectrum_idx]

                metrics = spectra_metrics_list[col_idx][gt_spectrum_idx] if has_metrics else None

                show_y_label = col_idx == 0
                show_x_label = col_idx == len(predicted_spectra_list) - 1

                spectrum_plot_settings = {
                    'top_peak_color': '#52B4FA',
                    #'top_peak_color': '120fd6',
                    'bottom_peak_color': '#E63434',
                    'max_mz': max_mz,
                    'show_x_label': show_x_label,
                    'show_y_label': show_y_label,
                    'show_annotation': show_annotation
                }

                cls._plot_spectrum_axes(ax, top_spectrum, bottom_spectrum, metrics, spectrum_plot_settings)

                pad = 5
                if col_idx == 0:
                    ax.annotate(top_spectrum.get_collision_energy_string(), xy=(0, 0.5),
                                xytext=(-ax.yaxis.labelpad - pad, 0),
                                xycoords=ax.yaxis.label, textcoords='offset points',
                                size=cls._default_title_font_size, weight='bold', ha='right', va='center')

            plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=3.0)
        return fig

    @classmethod
    def plot_mirrored_cfm_spectra(cls, groundtruth_spectra: CfmSpectra,
                                  predicted_spectra_list: List[CfmSpectra],
                                  show_predicted_spectra_info_text=True,
                                  show_mol_image_for_each_col=False,
                                  include_tanimoto_score=False,
                                  show_spectra_metric=True,
                                  show_top_mol_image=True,
                                  show_annotation=True):
        """Method to create figure to compare many predicted cfmid spectra vs  cfmid ground truth
           metrics include dice and dot product
        Args:
            groundtruth_spectra (CfmSpectra): [description]
            predicted_spectra_list (List[CfmSpectra]): [description]
            show_predicted_spectra_info_text (bool, optional): [description]. Defaults to True.
            show_mol_image_for_each_col (bool, optional): [description]. Defaults to False.
            show_top_mol_image (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        if show_spectra_metric:
            spectra_metrics_list, predicted_spectra_info_text = cls._compute_auto_cfm_metrics(groundtruth_spectra,
                                                                                              predicted_spectra_list,
                                                                                              include_tanimoto_score)
        else:
            spectra_metrics_list, predicted_spectra_info_text = None, None
        if show_predicted_spectra_info_text == False:
            predicted_spectra_info_text = None

        # if show_mol_image_for_each_col = False
        smiles_or_inchi = groundtruth_spectra.smiles_or_inchi if show_top_mol_image else None
        fig = cls.plot_mirrored_spectra(groundtruth_spectra.get_spectra_as_list(),
                                        [x.get_spectra_as_list() for x in predicted_spectra_list],
                                        [x.smiles_or_inchi for x in predicted_spectra_list],
                                        predicted_spectra_info_text,
                                        spectra_metrics_list,
                                        smiles_or_inchi,
                                        show_mol_image_for_each_col=show_mol_image_for_each_col,
                                        show_annotation=show_annotation)

        # plt.close(fig)
        return fig

    @classmethod
    def _compute_auto_cfm_metrics(cls, groundtruth_spectra: CfmSpectra, predicted_spectra_list: List[CfmSpectra],
                                  include_tanimoto_score=True):
        """[summary]

        Args:
            groundtruth_spectra (CfmSpectra): [description]
            predicted_spectra_list (List[CfmSpectra]): [description]

        Returns:
            [type]: spectra_metrics_list, predicted_spectra_info_text
        """
        spectra_metrics_list = [[] for _ in range(len(predicted_spectra_list))]
        predicted_spectra_info_text = ['' for _ in range(len(predicted_spectra_list))]
        mean_spectra_metrics_list = [{'Dice': 0, 'Dot Product': 0} for _ in range(len(predicted_spectra_list))]
        mum_spectra_per_sample = len(groundtruth_spectra.get_spectra_as_list())
        for idx, predicted_spectra in enumerate([x.get_spectra_as_list() for x in predicted_spectra_list]):
            for top_spectrum, bottom_spectrum in zip(groundtruth_spectra.get_spectra_as_list(), predicted_spectra):
                # matches = Comparators.get_matched_peaks(top_spectrum, bottom_spectrum, annotate = False)
                metrics = Comparators.get_metrics(top_spectrum, bottom_spectrum, verbose=False)
                # print(metrics)
                spectra_metrics_list[idx].append({'Dice': metrics['Dice'], 'Dot Product': metrics['Dot Product']})
                mean_spectra_metrics_list[idx]['Dice'] += metrics['Dice']
                # print(mean_spectra_metrics_list[idx]['Dice'])
                mean_spectra_metrics_list[idx]['Dot Product'] += metrics['Dot Product']
            predicted_spectra_info_text[idx] = 'Average Dice: {:.3f}\nAverage Dot Product: {:.3f}\n'.format(
                mean_spectra_metrics_list[idx]['Dice'] / mum_spectra_per_sample,
                mean_spectra_metrics_list[idx]['Dot Product'] / mum_spectra_per_sample)
            # print(predicted_spectra_info_text[idx])
        # print(mum_spectra_per_sample)
        # print('#'*50)
        if include_tanimoto_score:
            for idx, metrics in enumerate(mean_spectra_metrics_list):
                smiles_or_inchi = predicted_spectra_list[idx].smiles_or_inchi
                tanimoto_score = Utilities.get_tanimoto_score(groundtruth_spectra.smiles_or_inchi, smiles_or_inchi)
                predicted_spectra_info_text[idx] += 'Tanimoto Score: {:.3f}\n'.format(tanimoto_score)

        return spectra_metrics_list, predicted_spectra_info_text

    @classmethod
    def plot_identification_results(cls, predicted_spectra_smiles_or_inchi_list: List[str] = None,
                                    predicted_spectra_info_text: List[str] = None,
                                    smiles_or_inchi: str = None,
                                    info_text: str = None):
        """[summary]

        Args:
            predicted_spectra_smiles_or_inchi_list (List[str], optional): [description]. Defaults to None.
            predicted_spectra_info_text (List[str], optional): [description]. Defaults to None.
            smiles_or_inchi (str, optional): [description]. Defaults to None.
            info_text (str, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        num_predictions = len(predicted_spectra_smiles_or_inchi_list)
        num_col = 3 if num_predictions > 3 else num_predictions

        num_row = int(ceil(num_predictions / num_col))
        # print(num_row, num_col)

        height_ratios = [1] * num_row if smiles_or_inchi is None else [1] * (num_row + 1)
        fig, gs, cur_row = cls._init_figure_and_info_block(num_row, num_col, smiles_or_inchi, info_text,
                                                           height_scale=2.0, height_ratios=height_ratios)

        for idx, smiles_or_inchi in enumerate(predicted_spectra_smiles_or_inchi_list):
            # for row_idx in range(cur_row, cur_row + num_row):
            # print(idx//num_col + cur_row, idx % num_col)
            ax = fig.add_subplot(gs[idx // num_col + cur_row, idx % num_col])
            text_info = predicted_spectra_info_text[idx] if len(predicted_spectra_info_text) == num_predictions else ''
            cls._plot_info_block_axes(ax, smiles_or_inchi, info_text=text_info,
                                      info_block_font_size=cls._default_info_block_font_size)
        return fig

    @classmethod
    def plot_compare_cfm_identification_results(cls, groundtruth_spectra: CfmSpectra,
                                                predicted_spectra_dict: Dict[str, CfmSpectra],
                                                candidate_data_dict: Dict[str, List] = None,
                                                show_annotation: bool = False,
                                                use_cfm_color: bool = False):
        """[summary]

        Args:
            groundtruth_spectra (CfmSpectra): [description]
            predicted_spectra_list (List[CfmSpectra]): [description]
            extra_metrics (List[Dict[str, float]], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        num_spectrum_row = 3
        num_data_source = len(predicted_spectra_dict.values())
        info_block_font_size = 16

        # left side
        groundtruth_spectrum_list = groundtruth_spectra.get_spectra_as_list()

        # get max mz
        max_mz = 0
        for spectrum in groundtruth_spectrum_list:
            max_mz = max(spectrum.max_mz, max_mz)

        for data_source in predicted_spectra_dict:
            for spectrum in predicted_spectra_dict[data_source].get_spectra_as_list():
                max_mz = max(spectrum.max_mz, max_mz)
        max_mz += 50

        fig = plt.figure(figsize=(num_data_source * cls._default_ax_width + cls._default_ax_width,
                                  num_spectrum_row * cls._default_ax_height * 2), constrained_layout=True,
                         dpi=cls._default_dpi)
        grand_gs = GridSpec(ncols=3, nrows=1, figure=fig)
        left_gs = grand_gs[0:2].subgridspec(ncols=num_data_source, nrows=num_spectrum_row)
        col_idx = 0

        for data_source in predicted_spectra_dict:
            predicted_cfm_spectra = predicted_spectra_dict[data_source]
            predicted_spectra_list = predicted_cfm_spectra.get_spectra_as_list()
            row_idx = 0

            spectra_metrics_list, _ = cls._compute_auto_cfm_metrics(groundtruth_spectra,
                                                                    [predicted_spectra_dict[data_source]])
            for top_spectrum, bottom_spectrum, metrics in zip(groundtruth_spectrum_list, predicted_spectra_list,
                                                              spectra_metrics_list[0]):
                ax = fig.add_subplot(left_gs[row_idx, col_idx])

                # metrics = metrics
                show_y_label = col_idx == 0
                show_x_label = col_idx == len(predicted_spectra_list) - 1

                if use_cfm_color:
                    top_peak_color = cls._cfm_colors[0]
                    bottom_peak_color = cls._cfm_colors[1]
                else:
                    top_peak_color = cls._peak_colors[0]
                    bottom_peak_color = cls._peak_colors[col_idx % len(cls._peak_colors) + 1]

                spectrum_plot_settings = {
                    'top_peak_color': top_peak_color,
                    'bottom_peak_color': bottom_peak_color,
                    'max_mz': max_mz,
                    'show_x_label': show_x_label,
                    'show_y_label': show_y_label,
                    'show_annotation': show_annotation,
                    'show_legend': True,
                    'summary_font_size': info_block_font_size
                }

                cls._plot_spectrum_axes(ax, top_spectrum, bottom_spectrum,
                                        spectrum_plot_settings=spectrum_plot_settings)
                if row_idx == 0:
                    ax.text(0.0, 200, data_source + ' Spectra', fontsize=36)
                row_idx += 1

            col_idx += 1

        # right side
        # set 3 mols per column
        right_ncols = 3
        # set reuslts per datasource
        num_row_for_data_source = 1
        right_gs = grand_gs[2].subgridspec(ncols=right_ncols, nrows=num_row_for_data_source * num_data_source + 1)
        ax = fig.add_subplot(right_gs[0:right_ncols])
        text_info = '{}\n'.format(groundtruth_spectra.inchikey)

        target_mol_template = Utilities.create_mol_by_chemid(groundtruth_spectra.smiles_or_inchi)
        AllChem.Compute2DCoords(target_mol_template)

        cls._plot_info_block_axes(ax, groundtruth_spectra.smiles_or_inchi, 'Target Structure', text_info,
                                  info_block_font_size=info_block_font_size,
                                  info_block_title_font_size=info_block_font_size * 1.5,
                                  mol_template_coords=target_mol_template,
                                  image_width_resize_factor=3.5)
        ax.text(0.0, -20, 'Identification Results', fontsize=36)

        candidates_row_idx = 1
        for data_source in candidate_data_dict:
            candidates_list = candidate_data_dict[data_source]
            for candidate_idx, candidate in enumerate(candidates_list[0:right_ncols * num_row_for_data_source]):
                ax = fig.add_subplot(
                    right_gs[candidate_idx // right_ncols + candidates_row_idx, candidate_idx % right_ncols])
                candidate_smiles = candidate[2]
                tanimoto_score = Utilities.get_tanimoto_score(groundtruth_spectra.smiles_or_inchi, candidate_smiles)
                text_info = '{}'.format('Candidate #{}'.format(candidate_idx + 1))
                title_text = data_source + ' Identifications' if candidate_idx == 0 else ' '

                cls._plot_info_block_axes(ax, candidate_smiles, title_text,
                                          text_info, {"Ranking Score": candidate[0],
                                                      "Tanimoto": tanimoto_score},
                                          info_block_font_size=info_block_font_size,
                                          info_block_title_font_size=info_block_font_size * 1.5,
                                          mol_template_coords=target_mol_template)

            candidates_row_idx += num_row_for_data_source

        # fig.text(0.5, 0.01, 'm/z', ha='center')
        # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=10.0)
        return fig

    @classmethod
    def plot_cfm_spectra_with_fragmentation_annotation(cls, input_cfm_spectra: CfmSpectra):
        num_spectrum_row = 3
        info_block_font_size = 16

        cfm_spectra = deepcopy(input_cfm_spectra)
        # left side
        spectrum_list = cfm_spectra.get_spectra_as_list()

        # get max mz and fragments
        max_mz = 0
        max_fragement_count_per_grid = 24
        fragment_id_dict = {}
        fragment_structure_dict = {}
        fragment_structure_mz_dict = {}
        fragment_id = 1

        for spectrum in spectrum_list:
            max_mz = max(spectrum.max_mz, max_mz)
            for peak in spectrum.get_peaks():
                if peak.annotation not in fragment_id_dict:
                    fragment_id_dict[peak.annotation] = fragment_id
                    fragment_structure_dict[fragment_id] = peak.annotation
                    fragment_structure_mz_dict[fragment_id] = peak.mz
                    fragment_id += 1
                peak.annotation = fragment_id_dict[peak.annotation]
        num_fragments = len(fragment_structure_dict)

        max_mz = max_mz + 50
        # compute size of figure
        mun_fragment_grid = min(int(ceil(num_fragments / max_fragement_count_per_grid)), 3)
        fig = plt.figure(
            figsize=(cls._default_ax_width * (1 + mun_fragment_grid), num_spectrum_row * cls._default_ax_height * 2),
            constrained_layout=True, dpi=cls._default_dpi)

        grand_gs = GridSpec(ncols=1 + mun_fragment_grid, nrows=1, figure=fig)
        left_gs = grand_gs[0].subgridspec(ncols=1, nrows=num_spectrum_row)
        col_idx = 0

        row_idx = 0
        for spectrum in spectrum_list:
            ax = fig.add_subplot(left_gs[row_idx, col_idx])

            show_y_label = col_idx == 0
            show_x_label = col_idx == len(spectrum_list) - 1

            spectrum_plot_settings = {
                'top_peak_color': cls._peak_colors[0],
                'bottom_peak_color': cls._peak_colors[col_idx % len(cls._peak_colors) + 1],
                'max_mz': max_mz,
                'show_x_label': show_x_label,
                'show_y_label': show_y_label,
                'show_annotation': True,
                'show_legend': True
            }

            cls._plot_spectrum_axes(ax, spectrum, spectrum_plot_settings=spectrum_plot_settings)

            row_idx += 1
        col_idx += 1

        # right side

        grids = [
            [3, 5],  # 15
            [4, 4],  # 16
            [3, 6],  # 18
            [4, 5],  # 20
            [4, 6],  # 24

            [6, 5],  # 30
            [8, 4],  # 32
            [6, 6],  # 36
            [8, 5],  # 40
            [8, 6],  # 48

            [9, 5],  # 30
            [12, 4],  # 32
            [9, 6],  # 36
            [12, 5],  # 40
            [12, 6],  # 48
        ]

        for grid in grids:
            right_ncols, right_rows = grid
            if num_fragments <= grid[0] * grid[1]:
                break

        options = Draw.MolDrawOptions()
        options.bondLineWidth = max(2 * (right_rows / 6), 4)
        options.minFontSize = 16
        options.maxFontSize = 24
        options.noCarbonSymbols = True
        options.backgroundColour = None

        right_gs = grand_gs[1:].subgridspec(ncols=right_ncols, nrows=right_rows + 1)
        ax = fig.add_subplot(right_gs[0, 0])

        target_mol_template = Utilities.create_mol_by_chemid(cfm_spectra.smiles_or_inchi)
        AllChem.Compute2DCoords(target_mol_template)

        cls._plot_info_block_axes(ax, cfm_spectra.smiles_or_inchi,
                                  info_text='Input Structure  \n{}Da'.format(cfm_spectra.compound_exact_mass),
                                  info_block_font_size=info_block_font_size,
                                  mol_template_coords=target_mol_template,
                                  mol_image_options=options)

        ax.text(0, 1.0, 'Fragments', horizontalalignment='left',
                verticalalignment='center', fontsize=24, transform=ax.transAxes)

        candidates_row_idx = 1
        for fragment_id in range(0, right_ncols * right_rows):
            # print(fragment_id)
            if fragment_id + 1 in fragment_structure_dict:
                ax = fig.add_subplot(
                    right_gs[fragment_id // right_ncols + candidates_row_idx, fragment_id % right_ncols])
                candidate_smiles = fragment_structure_dict[fragment_id + 1]
                text_info = 'Fragment #{} \n{}Da'.format(fragment_id + 1, fragment_structure_mz_dict[fragment_id + 1])
                cls._plot_info_block_axes(ax, candidate_smiles, info_text=text_info,
                                          info_block_font_size=info_block_font_size,
                                          mol_template_coords=target_mol_template,
                                          mol_image_options=options)

        return fig

    @classmethod
    def plot_mirror_spectra_with_structure(cls, top_spectrum: Spectrum, bottom_spectrum: Spectrum,
                                           structure_names: List[str] = None, show_annotation=False,
                                           use_cfm_color=False, single_mol_image=False):

        info_block_font_size = 16

        # get max mz
        max_mz = max(top_spectrum.max_mz, bottom_spectrum.max_mz) + 50

        # init figure
        fig = plt.figure(figsize=(cls._default_ax_width * 2, cls._default_ax_height * 2), constrained_layout=True,
                         dpi=cls._default_dpi)

        gs = GridSpec(ncols=4, nrows=2, figure=fig)

        # top_spectrum.annotate_with_mz(intensity_threshold = annotate_intensity)
        # bottom_spectrum.annotate_with_mz(intensity_threshold = annotate_intensity)

        display_mol_template = Utilities.create_mol_by_chemid(top_spectrum.smiles_or_inchi)
        AllChem.Compute2DCoords(display_mol_template)

        if structure_names is None:
            structure_names = [None, None]

        if not single_mol_image:
            ax = fig.add_subplot(gs[0, 3])
            cls._plot_info_block_axes(ax, top_spectrum.smiles_or_inchi, structure_names[0],
                                      mol_template_coords=display_mol_template)

            ax = fig.add_subplot(gs[1, 3])
            cls._plot_info_block_axes(ax, bottom_spectrum.smiles_or_inchi, structure_names[1],
                                      mol_template_coords=display_mol_template)
        else:
            ax = fig.add_subplot(gs[0:2, 3])
            cls._plot_info_block_axes(ax, top_spectrum.smiles_or_inchi, structure_names[0],
                                      mol_template_coords=display_mol_template)

        show_y_label = True
        show_x_label = True

        ax = fig.add_subplot(gs[0:3, 0:3])
        if use_cfm_color:
            top_peak_color = cls._cfm_colors[0]
            bottom_peak_color = cls._cfm_colors[1]
        else:
            top_peak_color = cls._peak_colors[0]
            bottom_peak_color = cls._peak_colors[1]

        metrics = Comparators.get_metrics(top_spectrum, bottom_spectrum, verbose=False)

        spectrum_plot_settings = {
            'top_peak_color': top_peak_color,
            'bottom_peak_color': bottom_peak_color,
            'min_mz': -10,
            'max_mz': max_mz,
            'show_x_label': show_x_label,
            'show_y_label': show_y_label,
            'show_legend': True,
            'show_annotation': show_annotation,
            'summary_font_size': info_block_font_size,
            'text_type': None,
        }

        cls._plot_spectrum_axes(ax, top_spectrum, bottom_spectrum, metrics, spectrum_plot_settings)

        return fig