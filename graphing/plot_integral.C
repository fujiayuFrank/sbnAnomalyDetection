#include <regex>
#include <string>
#include <vector>
#include <iostream>
#include <set>

using namespace std;



// ------------------------------------------------------------
// Good/bad classification Switch
// true  = good runs one color, bad runs another color
// false = each run gets its own color
// ------------------------------------------------------------
bool color_by_good_bad = true;

// ------------------------------------------------------------
// List of run directories
// First one is the reference run
// ------------------------------------------------------------
vector<const char*> run_dirs = {
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_19305/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_19308/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_19315/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_829/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20769/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20782/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20768/reco/",
    
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20614/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20615/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20620/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20621/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20173/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_830/reco/"
    // Add more directories here if needed:

};

// ------------------------------------------------------------
// Good/bad classification Sets
// ------------------------------------------------------------

set<int> good_runs = {
    19305, 19308, 19315, 829, 20769, 20782, 20768
};

set<int> bad_runs = {
    20614, 20615, 20620, 20621, 20173, 830
};

int good_color = kBlue;
int bad_color = kRed;
int unknown_color = kBlack;


// ------------------------------------------------------------
// Extract run number from path
// ------------------------------------------------------------

int extract_run_number(const char* path) {
    std::string s(path);

    std::regex pattern("CI_build_lar_ci_([0-9]+)");
    std::smatch match;

    if (std::regex_search(s, match, pattern)) {
        return std::stoi(match[1]);
    }

    return -1;
}


// ------------------------------------------------------------
// Decide color from good/bad label
// ------------------------------------------------------------

int get_run_color(int run) {
    if (good_runs.count(run)) return good_color;
    if (bad_runs.count(run)) return bad_color;
    return unknown_color;
}


// ------------------------------------------------------------
// Decide label from good/bad label
// ------------------------------------------------------------

const char* get_run_status(int run) {
    if (good_runs.count(run)) return "good";
    if (bad_runs.count(run)) return "bad";
    return "unknown";
}


// ------------------------------------------------------------
// Main plotting function
// ------------------------------------------------------------

void plot_integral() {
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);

    const char* treePath = "caloskim/TrackCaloSkim";
    const char* branch   = "hits2.h.integral";

    int nruns = run_dirs.size();

    if (nruns < 2) {
        cout << "Need at least two run directories." << endl;
        return;
    }

    // --------------------------------------------------------
    // Histogram settings
    // --------------------------------------------------------

    int nbins = 100;
    double xmin = 0;
    double xmax = 4000;

    vector<int> run_numbers;
    vector<TChain*> chains;
    vector<TH1D*> hists;
    vector<TH1D*> bands;

    // --------------------------------------------------------
    // Build chains and histograms
    // --------------------------------------------------------

    for (int i = 0; i < nruns; i++) {
        const char* dir = run_dirs[i];
        int run = extract_run_number(dir);
        run_numbers.push_back(run);

        TChain* chain = new TChain(treePath);

        TString pattern = Form("%s/DQMValidationTrees_*.root", dir);
        int nfiles = chain->Add(pattern);

        // ROOT I/O cache
        chain->SetCacheSize(100 * 1024 * 1024);
        chain->AddBranchToCache(branch, kTRUE);

        cout << "Run " << run << ": added " << nfiles << " ROOT files" << endl;

        if (nfiles == 0) {
            cout << "No DQMValidationTrees ROOT files found in directory: " << dir << endl;
            return;
        }

        cout << "Run " << run << " entries = " << chain->GetEntries() << endl;

        chains.push_back(chain);

        TH1D* h = new TH1D(
            Form("h_run_%d", run),
            "Hit Integral Comparison;Integral;Hits",
            nbins,
            xmin,
            xmax
        );

        h->Sumw2();

        chain->Draw(Form("%s >> h_run_%d", branch, run), "", "goff");

        cout << "Raw integral run " << run << " = " << h->Integral() << endl;

        hists.push_back(h);
    }

    // --------------------------------------------------------
    // Normalize all non-reference runs to the reference run
    // --------------------------------------------------------

    TH1D* h_ref = hists[0];
    int run_ref = run_numbers[0];

    double ref_integral = h_ref->Integral();

    for (int i = 1; i < nruns; i++) {
        double integral = hists[i]->Integral();

        if (integral > 0) {
            double scale = ref_integral / integral;
            hists[i]->Scale(scale);

            cout << "Scale factor applied to run "
                 << run_numbers[i]
                 << " = "
                 << scale
                 << endl;
        }
    }

    // --------------------------------------------------------
    // Calculate chi-square / ndf for each run vs reference
    // --------------------------------------------------------

    vector<double> chi2_values;
    vector<int> ndf_values;
    vector<double> reduced_chi2_values;

    chi2_values.push_back(0.0);
    ndf_values.push_back(0);
    reduced_chi2_values.push_back(0.0);

    for (int j = 1; j < nruns; j++) {
        double chi2 = 0.0;
        int n_used_bins = 0;

        TH1D* h_other = hists[j];

        for (int i = 1; i <= nbins; i++) {
            double a  = h_ref->GetBinContent(i);
            double ea = h_ref->GetBinError(i);

            double b  = h_other->GetBinContent(i);
            double eb = h_other->GetBinError(i);

            double err2 = ea * ea + eb * eb;

            if (err2 > 0 && (a > 0 || b > 0)) {
                chi2 += (a - b) * (a - b) / err2;
                n_used_bins++;
            }
        }

        int ndf = n_used_bins - 1;
        double reduced_chi2 = 0.0;

        if (ndf > 0) {
            reduced_chi2 = chi2 / ndf;
        }

        chi2_values.push_back(chi2);
        ndf_values.push_back(ndf);
        reduced_chi2_values.push_back(reduced_chi2);

        cout << endl;
        cout << "Comparison: run " << run_ref << " vs run " << run_numbers[j] << endl;
        cout << "Chi2 = " << chi2 << endl;
        cout << "NDF = " << ndf << endl;
        cout << "Reduced chi2 = " << reduced_chi2 << endl;
    }

    // --------------------------------------------------------
    // Colors and styles
    // --------------------------------------------------------

    vector<int> colors = {
        kBlack,
        kRed,
        kBlue,
        kGreen + 2,
        kOrange + 7,
        kMagenta,
        kCyan + 2,
        kViolet,
        kBrown
    };

    vector<int> marker_styles = {
        20,
        24,
        21,
        25,
        22,
        26,
        23,
        32,
        33
    };

    for (int i = 0; i < nruns; i++) {
        int color;

        if (color_by_good_bad) {
            color = get_run_color(run_numbers[i]);
        } else {
            color = colors[i % colors.size()];
        }

        int marker = marker_styles[i % marker_styles.size()];

        hists[i]->SetLineColor(color);
        hists[i]->SetLineWidth(2);
        hists[i]->SetMarkerColor(color);
        hists[i]->SetMarkerStyle(marker);
        hists[i]->SetMarkerSize(0.6);

        TH1D* band = (TH1D*)hists[i]->Clone(Form("band_run_%d", run_numbers[i]));
        band->SetFillColorAlpha(color, 0.18);
        band->SetFillStyle(1001);
        band->SetLineColor(color);
        band->SetMarkerSize(0);

        bands.push_back(band);
    }

    // --------------------------------------------------------
    // Canvas and pads
    // --------------------------------------------------------

    TCanvas* c = new TCanvas("c", "comparison", 1200, 900);

    TPad* pad1 = new TPad("pad1", "top", 0.0, 0.32, 1.0, 1.0);
    TPad* pad2 = new TPad("pad2", "bottom", 0.0, 0.0, 1.0, 0.32);

    pad1->SetBottomMargin(0.02);
    pad1->SetGridx();
    pad1->SetGridy();

    pad2->SetTopMargin(0.04);
    pad2->SetBottomMargin(0.30);
    pad2->SetGridx();
    pad2->SetGridy();

    pad1->Draw();
    pad2->Draw();

    // --------------------------------------------------------
    // Top plot
    // --------------------------------------------------------

    pad1->cd();

    double ymax = 0.0;

    for (int j = 0; j < nruns; j++) {
        for (int i = 1; i <= nbins; i++) {
            ymax = TMath::Max(
                ymax,
                hists[j]->GetBinContent(i) + hists[j]->GetBinError(i)
            );
        }
    }

    bands[0]->SetTitle("Integral of gaussian fit to ADC values Collection;Integral Value;Hits");
    bands[0]->SetMaximum(1.15 * ymax);
    bands[0]->SetMinimum(0);

    bands[0]->GetXaxis()->SetLabelSize(0);
    bands[0]->GetXaxis()->SetTitleSize(0);

    bands[0]->GetYaxis()->SetTitle("Hits");
    bands[0]->GetYaxis()->SetTitleSize(0.05);
    bands[0]->GetYaxis()->SetLabelSize(0.045);

    // Draw error bands first
    bands[0]->Draw("E2");

    for (int i = 1; i < nruns; i++) {
        bands[i]->Draw("E2 SAME");
    }

    // Draw histograms and points
    for (int i = 0; i < nruns; i++) {
        hists[i]->Draw("HIST SAME");
        hists[i]->Draw("E1 SAME");
    }

    gPad->RedrawAxis();

    TLegend* leg = new TLegend(0.56, 0.55, 0.88, 0.88);
    leg->SetBorderSize(1);
    leg->SetFillColor(kWhite);

    for (int i = 0; i < nruns; i++) {
        if (i == 0) {
            leg->AddEntry(
                hists[i],
                Form("Data ID %d reference (%s)", run_numbers[i], get_run_status(run_numbers[i])),
                "lep"
            );
        } else {
            leg->AddEntry(
                hists[i],
                Form("Data ID %d normalized (%s)", run_numbers[i], get_run_status(run_numbers[i])),
                "lep"
            );
        }
    }

    leg->Draw();

    // Chi-square text
    TLatex chi2_text;
    chi2_text.SetNDC();
    chi2_text.SetTextSize(0.030);

    double y_text = 0.50;

    for (int i = 1; i < nruns; i++) {
        chi2_text.DrawLatex(
            0.58,
            y_text,
            Form("run %d: #chi^{2}/ndf = %.6f",
                 run_numbers[i],
                 reduced_chi2_values[i])
        );

        y_text -= 0.04;
    }

    // --------------------------------------------------------
    // Bottom plot: fractional differences
    // (reference - other run) / reference
    // --------------------------------------------------------

    pad2->cd();

    vector<TH1D*> ratios;

    for (int j = 1; j < nruns; j++) {
        TH1D* ratio = (TH1D*)h_ref->Clone(
            Form("ratio_run_%d", run_numbers[j])
        );

        ratio->Reset();
        ratio->SetTitle("");

        TH1D* h_other = hists[j];

        for (int i = 1; i <= nbins; i++) {
            double a  = h_ref->GetBinContent(i);
            double ea = h_ref->GetBinError(i);

            double b  = h_other->GetBinContent(i);
            double eb = h_other->GetBinError(i);

            if (a > 0) {
                double r = (a - b) / a;

                // f = (a - b)/a = 1 - b/a
                // df/da = b/a^2
                // df/db = -1/a
                double er = sqrt(
                    (b / (a * a) * ea) * (b / (a * a) * ea)
                    + (eb / a) * (eb / a)
                );

                ratio->SetBinContent(i, r);
                ratio->SetBinError(i, er);
            } else {
                ratio->SetBinContent(i, 0);
                ratio->SetBinError(i, 0);
            }
        }

        int color;

        if (color_by_good_bad) {
            color = get_run_color(run_numbers[j]);
        } else {
            color = colors[j % colors.size()];
        }

        int marker = marker_styles[j % marker_styles.size()];

        ratio->SetMarkerStyle(marker);
        ratio->SetMarkerSize(0.6);
        ratio->SetMarkerColor(color);
        ratio->SetLineColor(color);

        ratios.push_back(ratio);
    }

    // First ratio creates axes
    ratios[0]->GetYaxis()->SetTitle(
        Form("#frac{(run %d) - (other run)}{(run %d)}",
             run_ref,
             run_ref)
    );

    ratios[0]->GetXaxis()->SetTitle("Integral");

    ratios[0]->GetYaxis()->SetRangeUser(-1.0, 1.0);
    ratios[0]->GetYaxis()->SetNdivisions(505);

    ratios[0]->GetYaxis()->SetTitleSize(0.08);
    ratios[0]->GetYaxis()->SetLabelSize(0.075);
    ratios[0]->GetYaxis()->SetTitleOffset(0.55);

    ratios[0]->GetXaxis()->SetTitleSize(0.10);
    ratios[0]->GetXaxis()->SetLabelSize(0.08);
    ratios[0]->GetXaxis()->SetTitleOffset(1.0);

    ratios[0]->Draw("E1");

    for (int i = 1; i < ratios.size(); i++) {
        ratios[i]->Draw("E1 SAME");
    }

    TLine* line = new TLine(xmin, 0.0, xmax, 0.0);
    line->SetLineColor(kBlack);
    line->SetLineWidth(2);
    line->Draw("SAME");

    TLegend* ratio_leg = new TLegend(0.68, 0.68, 0.88, 0.88);
    ratio_leg->SetBorderSize(1);
    ratio_leg->SetFillColor(kWhite);

    for (int j = 1; j < nruns; j++) {
        ratio_leg->AddEntry(
            ratios[j - 1],
            Form("run %d (%s)", run_numbers[j], get_run_status(run_numbers[j])),
            "lep"
        );
    }

    ratio_leg->Draw();

    // --------------------------------------------------------
    // Save output
    // --------------------------------------------------------

    TString run_tag = "";

    for (int i = 0; i < nruns; i++) {
        if (i > 0) run_tag += "_";
        run_tag += Form("%d", run_numbers[i]);
    }

    TString mode_tag = color_by_good_bad ? "good_bad_colors" : "multi_colors";

    c->SaveAs(Form("integral_comparison_many_runs_%s_%s.png", mode_tag.Data(), run_tag.Data()));
    c->SaveAs(Form("integral_comparison_many_runs_%s_%s.pdf", mode_tag.Data(), run_tag.Data()));

    cout << "Saved "
         << Form("integral_comparison_many_runs_%s_%s.png", mode_tag.Data(), run_tag.Data())
         << " and "
         << Form("integral_comparison_many_runs_%s_%s.pdf", mode_tag.Data(), run_tag.Data())
         << endl;
}