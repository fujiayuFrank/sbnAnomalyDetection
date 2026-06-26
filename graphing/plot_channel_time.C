#include <regex>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <set>
#include <utility>

using namespace std;


// ------------------------------------------------------------
// Multi-run channel integral vs time plotter
// ------------------------------------------------------------
//
// Purpose:
//   Read DQMValidationTrees_*.root from multiple run directories.
//   For selected channels, plot:
//       y-axis = hits2.h.integral
//       x-axis = meta.time, usually shifted to time since global first event [days]
//
//   Color meaning:
//       good run = cold color palette
//       bad run  = hot color palette
//       unknown  = black
//
//   Marker shape meaning:
//       channel
//
//   Legend label format:
//       run-19305-channel5353-good
//
// Usage:
//   root -l -q -b 'channel_time_plot.C("channel_time_plot()")'
//
// Or inside ROOT:
//   .L channel_time_plot.C
//   channel_time_plot()
//
// ------------------------------------------------------------


// ------------------------------------------------------------
// User settings
// ------------------------------------------------------------

vector<const char*> RUN_DIRS = {
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_19305/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_19308/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20769/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20782/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20768/reco/",

    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20614/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20615/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20620/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_20173/reco/",
    "/pnfs/icarus/persistent/users/micarrig/DQM/CI_build_lar_ci_830/reco/",
};


// Channels to plot.
vector<int> CHANNEL_LIST = {
    9879,
    // 5535,
    // Add more channels here.
};


// meta.time appears to be nanoseconds-scale in your files.
// 1 day = 24 * 60 * 60 * 1e9 ns.
double DAY_WIDTH = 86400.0 * 1.0e9;


// Easier x-axis: time offset in days.
// If ALIGN_RUNS_BY_OWN_START = true, every run starts at x = 0.
// This overlays runs by relative time instead of their real calendar time.
bool USE_TIME_OFFSET_IN_DAYS = true;
bool ALIGN_RUNS_BY_OWN_START = true;


// Set true if you want raw meta.time on the x-axis.
bool USE_RAW_META_TIME = false;


// Draw lines + points.
// If false, points only.
bool CONNECT_POINTS = true;


// Optional y-axis range.
bool USE_INTEGRAL_Y_RANGE = false;
double INTEGRAL_Y_MIN = 0.0;
double INTEGRAL_Y_MAX = 4000.0;


// Use -1 for all files. Use a small number for testing.
int MAX_FILES_PER_RUN = -1;


// Optional skip list for files that hang before TFile::Open finishes.
vector<TString> SKIP_FILE_PATTERNS = {
    // "DQMValidationTrees_011.root",
    // "DQMValidationTrees_11.root"
};


// ------------------------------------------------------------
// ROOT tree / branch names
// ------------------------------------------------------------

const char* TREE_PATH = "caloskim/TrackCaloSkim";
const char* TIME_BRANCH = "meta.time";
const char* INTEGRAL_BRANCH = "hits2.h.integral";
const char* CHANNEL_BRANCH = "hits2.h.channel";


// ------------------------------------------------------------
// Good/bad classification
// ------------------------------------------------------------

set<int> GOOD_RUNS = {
    19305, 19308, 19315, 829, 20769, 20782, 20768
};

set<int> BAD_RUNS = {
    20614, 20615, 20620, 20621, 20173, 830
};


// Good runs use cold colors.
vector<int> GOOD_COLORS = {
    TColor::GetColor("#2483c8"),
    TColor::GetColor("#063d6b"),
    TColor::GetColor("#17becf"),
    TColor::GetColor("#00a087"),
    TColor::GetColor("#2ca02c"),
    TColor::GetColor("#4daf4a"),
    TColor::GetColor("#66c2a5")
};


// Bad runs use hot colors.
vector<int> BAD_COLORS = {
    TColor::GetColor("#d62728"),
    TColor::GetColor("#e41a1c"),
    TColor::GetColor("#b2182b"),
    TColor::GetColor("#ff7f0e"),
    TColor::GetColor("#a65628"),
    TColor::GetColor("#613807")
};


// Marker shape means channel.
vector<int> MARKER_STYLES = {
    20, // filled circle
    21, // filled square
    22, // filled triangle up
    23, // filled triangle down
    24, // open circle
    25, // open square
    26, // open triangle up
    27, // open diamond
    28, // open cross
    33, // filled diamond
    34  // filled cross
};


// ------------------------------------------------------------
// Helper: extract run number from path
// ------------------------------------------------------------

int extract_run_number(const char* path) {
    string s(path);
    regex pattern("CI_build_lar_ci_([0-9]+)");
    smatch match;

    if (regex_search(s, match, pattern)) {
        return stoi(match[1]);
    }

    return -1;
}


const char* get_run_status(int run) {
    if (GOOD_RUNS.count(run)) return "good";
    if (BAD_RUNS.count(run)) return "bad";
    return "unknown";
}


int get_run_color(int run) {
    if (GOOD_RUNS.count(run)) {
        int index = 0;

        for (const char* dir : RUN_DIRS) {
            int r = extract_run_number(dir);

            if (!GOOD_RUNS.count(r)) continue;

            if (r == run) {
                return GOOD_COLORS[index % GOOD_COLORS.size()];
            }

            index++;
        }
    }

    if (BAD_RUNS.count(run)) {
        int index = 0;

        for (const char* dir : RUN_DIRS) {
            int r = extract_run_number(dir);

            if (!BAD_RUNS.count(r)) continue;

            if (r == run) {
                return BAD_COLORS[index % BAD_COLORS.size()];
            }

            index++;
        }
    }

    return kBlack;
}


bool should_skip_file(const TString& file_path) {
    for (const TString& pat : SKIP_FILE_PATTERNS) {
        if (pat.Length() > 0 && file_path.Contains(pat)) {
            return true;
        }
    }

    return false;
}


// ------------------------------------------------------------
// Add readable ROOT files to one TChain
// ------------------------------------------------------------

int add_files_to_chain(TChain* chain, const char* dir) {
    TString command = Form("ls %s/DQMValidationTrees_*.root 2>/dev/null", dir);
    TString file_list = gSystem->GetFromPipe(command);

    TObjArray* lines = file_list.Tokenize("\n");
    int n_added = 0;

    for (int i = 0; i < lines->GetEntries(); i++) {
        if (MAX_FILES_PER_RUN > 0 && n_added >= MAX_FILES_PER_RUN) break;

        TString file_path = lines->At(i)->GetName();

        if (file_path.Length() == 0) continue;

        if (should_skip_file(file_path)) {
            cout << "Skipping known troublesome file: " << file_path << endl;
            continue;
        }

        TFile* f = TFile::Open(file_path);

        if (!f || f->IsZombie()) {
            cout << "Skipping bad/unreadable file: " << file_path << endl;
            if (f) f->Close();
            continue;
        }

        TTree* t = (TTree*)f->Get(TREE_PATH);

        if (!t) {
            cout << "Skipping file without tree " << TREE_PATH << ": "
                 << file_path << endl;
            f->Close();
            continue;
        }

        f->Close();

        chain->Add(file_path);
        n_added++;

        cout << "Added file " << n_added << ": " << file_path << endl;
    }

    delete lines;

    return n_added;
}


// ------------------------------------------------------------
// Find min and max meta.time over events in one chain
// ------------------------------------------------------------

bool find_time_range(TChain* chain, double& t_min, double& t_max) {
    Long64_t n = chain->Draw(TIME_BRANCH, "", "goff");

    if (n <= 0) {
        cout << "Could not read any meta.time values." << endl;
        return false;
    }

    double* times = chain->GetV1();

    bool found = false;

    for (Long64_t i = 0; i < n; i++) {
        double t = times[i];

        if (!TMath::Finite(t)) continue;

        if (!found) {
            t_min = t;
            t_max = t;
            found = true;
        } else {
            if (t < t_min) t_min = t;
            if (t > t_max) t_max = t;
        }
    }

    return found;
}


// ------------------------------------------------------------
// Main plotting function
// ------------------------------------------------------------

void plot_integral_vs_time_channels_multi() {
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);

    if (RUN_DIRS.empty()) {
        cout << "RUN_DIRS is empty." << endl;
        return;
    }

    if (CHANNEL_LIST.empty()) {
        cout << "CHANNEL_LIST is empty." << endl;
        return;
    }

    vector<TChain*> chains;
    vector<int> run_numbers;
    vector<int> nfiles_per_run;
    vector<double> run_t_mins;
    vector<double> run_t_maxs;

    double global_t_min = 0.0;
    double global_t_max = 0.0;
    bool first_time_range = true;

    // --------------------------------------------------------
    // Build one chain per run directory and find global time range
    // --------------------------------------------------------

    for (const char* dir : RUN_DIRS) {
        int run = extract_run_number(dir);

        cout << endl;
        cout << "============================================================" << endl;
        cout << "Run directory: " << dir << endl;
        cout << "Run number: " << run << " (" << get_run_status(run) << ")" << endl;
        cout << "============================================================" << endl;

        TChain* chain = new TChain(TREE_PATH);

        int nfiles = add_files_to_chain(chain, dir);

        if (nfiles == 0) {
            cout << "No readable ROOT files were added for run " << run << endl;
            delete chain;
            continue;
        }

        chain->SetCacheSize(100 * 1024 * 1024);
        chain->AddBranchToCache(TIME_BRANCH, kTRUE);
        chain->AddBranchToCache(INTEGRAL_BRANCH, kTRUE);
        chain->AddBranchToCache(CHANNEL_BRANCH, kTRUE);

        cout << "Run " << run << ": total files added = " << nfiles << endl;
        cout << "Run " << run << ": total entries = " << chain->GetEntries() << endl;

        double t_min = 0.0;
        double t_max = 0.0;

        if (!find_time_range(chain, t_min, t_max)) {
            cout << "Could not determine time range for run " << run << endl;
            delete chain;
            continue;
        }

        cout << scientific << setprecision(17);
        cout << "Run " << run << " meta.time min = " << t_min << endl;
        cout << "Run " << run << " meta.time max = " << t_max << endl;
        cout << defaultfloat;

        if (first_time_range) {
            global_t_min = t_min;
            global_t_max = t_max;
            first_time_range = false;
        } else {
            if (t_min < global_t_min) global_t_min = t_min;
            if (t_max > global_t_max) global_t_max = t_max;
        }

        chains.push_back(chain);
        run_numbers.push_back(run);
        nfiles_per_run.push_back(nfiles);
        run_t_mins.push_back(t_min);
        run_t_maxs.push_back(t_max);
    }

    if (chains.empty()) {
        cout << "No usable chains were built." << endl;
        return;
    }

    cout << endl;
    cout << "================ Global time range ================" << endl;
    cout << scientific << setprecision(17);
    cout << "global meta.time min = " << global_t_min << endl;
    cout << "global meta.time max = " << global_t_max << endl;
    cout << "global time span in days = "
         << (global_t_max - global_t_min) / DAY_WIDTH
         << endl;

    double max_run_span_days = 0.0;

    for (int i = 0; i < (int)run_t_mins.size(); i++) {
        double span_days = (run_t_maxs[i] - run_t_mins[i]) / DAY_WIDTH;
        if (span_days > max_run_span_days) max_run_span_days = span_days;
    }

    cout << "max individual run time span in days = "
         << max_run_span_days
         << endl;

    if (ALIGN_RUNS_BY_OWN_START) {
        cout << "Runs will be aligned by their own first event time." << endl;
    } else {
        cout << "Runs will use the global first event time." << endl;
    }

    cout << defaultfloat;

    // --------------------------------------------------------
    // Canvas and legend
    // --------------------------------------------------------

    TCanvas* c = new TCanvas(
        "c_integral_vs_time_multi_runs",
        "hits2.h.integral vs meta.time for selected channels, multiple runs",
        1700,
        900
    );

    c->SetGridx();
    c->SetGridy();

    // Leave room on right for legend.
    c->SetRightMargin(0.32);

    TMultiGraph* mg = new TMultiGraph();

    TLegend* leg = new TLegend(0.70, 0.12, 0.98, 0.88);
    leg->SetBorderSize(1);
    leg->SetFillColor(kWhite);
    leg->SetTextSize(0.017);
    leg->SetNColumns(1);

    set<string> added_legend_labels;

    vector<TGraph*> graphs;

    double global_y_min = 0.0;
    double global_y_max = 0.0;
    bool first_point = true;

    // --------------------------------------------------------
    // For each run and channel, make one graph
    // No day coloring, no day splitting.
    // Color = run good/bad palette.
    // Marker = channel.
    // --------------------------------------------------------

    for (int r_index = 0; r_index < (int)chains.size(); r_index++) {
        TChain* chain = chains[r_index];
        int run = run_numbers[r_index];

        int run_color = get_run_color(run);
        const char* status = get_run_status(run);

        for (int ch_index = 0; ch_index < (int)CHANNEL_LIST.size(); ch_index++) {
            int channel = CHANNEL_LIST[ch_index];

            TString x_expr;

            double this_t0 = ALIGN_RUNS_BY_OWN_START ? run_t_mins[r_index] : global_t_min;

            if (USE_RAW_META_TIME) {
                x_expr = TIME_BRANCH;
            } else if (USE_TIME_OFFSET_IN_DAYS) {
                x_expr = Form("(%s - %.17e) / %.17e", TIME_BRANCH, this_t0, DAY_WIDTH);
            } else {
                x_expr = Form("%s - %.17e", TIME_BRANCH, this_t0);
            }

            TString draw_expr = Form("%s:%s", INTEGRAL_BRANCH, x_expr.Data());

            TString cut = Form(
                "%s == %d",
                CHANNEL_BRANCH,
                channel
            );

            Long64_t npoints = chain->Draw(draw_expr, cut, "goff");

            cout << "run " << run
                 << " (" << status << ")"
                 << ", channel " << channel
                 << ": npoints = " << npoints
                 << endl;

            if (npoints <= 0) continue;

            double* y_raw = chain->GetV1(); // hits2.h.integral
            double* x_raw = chain->GetV2(); // time expression

            vector<pair<double, double>> points;

            for (Long64_t i = 0; i < npoints; i++) {
                double x = x_raw[i];
                double y = y_raw[i];

                if (!TMath::Finite(x)) continue;
                if (!TMath::Finite(y)) continue;

                points.push_back({x, y});
            }

            sort(points.begin(), points.end());

            if (points.size() < 1) continue;

            vector<double> xs;
            vector<double> ys;

            xs.reserve(points.size());
            ys.reserve(points.size());

            for (const auto& p : points) {
                xs.push_back(p.first);
                ys.push_back(p.second);
            }

            TGraph* gr = new TGraph(xs.size(), xs.data(), ys.data());

            int marker = MARKER_STYLES[ch_index % MARKER_STYLES.size()];

            gr->SetName(Form("gr_run_%d_channel_%d", run, channel));
            gr->SetTitle(Form("run-%d-channel%d-%s", run, channel, status));

            gr->SetMarkerColor(run_color);
            gr->SetLineColor(run_color);
            gr->SetLineWidth(2);

            // Marker shape by channel.
            gr->SetMarkerStyle(marker);
            gr->SetMarkerSize(1.0);

            TString draw_opt = CONNECT_POINTS ? "LP" : "P";

            mg->Add(gr, draw_opt);
            graphs.push_back(gr);

            TString label = Form("run-%d-channel%d-%s", run, channel, status);
            string label_str = label.Data();

            if (added_legend_labels.count(label_str) == 0) {
                leg->AddEntry(gr, label, CONNECT_POINTS ? "lp" : "p");
                added_legend_labels.insert(label_str);
            }

            for (double y : ys) {
                if (first_point) {
                    global_y_min = y;
                    global_y_max = y;
                    first_point = false;
                } else {
                    if (y < global_y_min) global_y_min = y;
                    if (y > global_y_max) global_y_max = y;
                }
            }
        }
    }

    if (graphs.empty()) {
        cout << "No matching points found for requested runs/channels." << endl;
        return;
    }

    // --------------------------------------------------------
    // Draw
    // --------------------------------------------------------

    TString graph_title;

    if (USE_RAW_META_TIME) {
        graph_title = "hits2.h.integral vs meta.time;meta.time;hits2.h.integral";
    } else if (USE_TIME_OFFSET_IN_DAYS) {
        if (ALIGN_RUNS_BY_OWN_START) {
            graph_title = "hits2.h.integral vs meta.time;Time since each run first event [days];hits2.h.integral";
        } else {
            graph_title = "hits2.h.integral vs meta.time;Time since global first event [days];hits2.h.integral";
        }
    } else {
        if (ALIGN_RUNS_BY_OWN_START) {
            graph_title = "hits2.h.integral vs meta.time;meta.time - each run first meta.time;hits2.h.integral";
        } else {
            graph_title = "hits2.h.integral vs meta.time;meta.time - global first meta.time;hits2.h.integral";
        }
    }

    mg->SetTitle(graph_title);

    TString mg_draw_opt = CONNECT_POINTS ? "ALP" : "AP";
    mg->Draw(mg_draw_opt);

    if (USE_INTEGRAL_Y_RANGE) {
        mg->GetYaxis()->SetRangeUser(INTEGRAL_Y_MIN, INTEGRAL_Y_MAX);
    } else {
        double margin = 0.10 * (global_y_max - global_y_min);
        if (margin <= 0) margin = 1.0;
        mg->GetYaxis()->SetRangeUser(global_y_min - margin, global_y_max + margin);
    }

    mg->GetXaxis()->SetTitleOffset(1.2);
    mg->GetYaxis()->SetTitleOffset(1.3);

    leg->Draw();
    c->Update();

    // --------------------------------------------------------
    // Save
    // --------------------------------------------------------

    TString run_tag = "";

    for (int i = 0; i < (int)run_numbers.size(); i++) {
        if (i > 0) run_tag += "_";
        run_tag += Form("%d", run_numbers[i]);
    }

    TString channel_tag = "";

    for (int i = 0; i < (int)CHANNEL_LIST.size(); i++) {
        if (i > 0) channel_tag += "_";
        channel_tag += Form("%d", CHANNEL_LIST[i]);
    }

    TString x_tag;

    if (USE_RAW_META_TIME) {
        x_tag = "rawtime";
    } else if (ALIGN_RUNS_BY_OWN_START) {
        x_tag = "aligned_run_start_days";
    } else {
        x_tag = "global_timeoffsetdays";
    }

    TString png_name = Form(
        "integral_vs_time_multi_runs_%s_channels_%s_%s.png",
        run_tag.Data(),
        channel_tag.Data(),
        x_tag.Data()
    );

    TString pdf_name = Form(
        "integral_vs_time_multi_runs_%s_channels_%s_%s.pdf",
        run_tag.Data(),
        channel_tag.Data(),
        x_tag.Data()
    );

    c->SaveAs(png_name);
    c->SaveAs(pdf_name);

    cout << "Saved " << png_name << " and " << pdf_name << endl;
}


// ------------------------------------------------------------
// Wrapper
// If the file is named channel_time_plot_multi.C, this runs automatically.
// ------------------------------------------------------------

void channel_time_plot() {
    plot_integral_vs_time_channels_multi();
}
