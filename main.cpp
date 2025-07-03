#include "EnhancedReservoirSimulator.h"
#include <iostream>
#include <string>
#include <exception>
#include <fstream>
#include <sstream>  // For std::stringstream
#include <iomanip>
#include <chrono>
#include <memory>
#include <csignal>
#include <cstring>  // For memset
#include <algorithm>  // For std::min_element, std::max_element
#include <ctime>  // For time functions
#include <iterator>  // For std::istreambuf_iterator

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
#ifndef NO_WIN_DIALOGS
#include <commdlg.h>
#endif
#include <direct.h>  // For _mkdir
#else
#include <sys/stat.h>
#include <sys/types.h>
#ifndef MAX_PATH
#define MAX_PATH 260
#endif
#endif

// Global simulator pointer for signal handling
std::unique_ptr<EnhancedReservoirSimulator> g_simulator;

// Signal handler for graceful shutdown
void signalHandler(int signum) {
    std::cout << "\n\nReceived signal " << signum << ". Shutting down gracefully..." << std::endl;
    if (g_simulator) {
        g_simulator->stop();
    }
    exit(signum);
}

class SimulationManager {
private:
    std::string input_file_;
    std::string output_dir_;
    bool verbose_;
    bool benchmark_mode_;
    
public:
    SimulationManager() : verbose_(false), benchmark_mode_(false) {}
    
    /**
     * @brief Open file dialog to select input file
     */
    std::string openFileDialog() {
#ifdef _WIN32
#ifndef NO_WIN_DIALOGS
        char filename[MAX_PATH] = "";
        
        OPENFILENAMEA ofn;
        memset(&ofn, 0, sizeof(ofn));
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = NULL;
        ofn.lpstrFilter = "Text Files (*.txt)\0*.txt\0Data Files (*.dat)\0*.dat\0All Files (*.*)\0*.*\0";
        ofn.lpstrFile = filename;
        ofn.nMaxFile = MAX_PATH;
        ofn.lpstrTitle = "Select Reservoir Simulation Input File";
        ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;
        ofn.lpstrDefExt = "txt";
        
        if (GetOpenFileNameA(&ofn)) {
            return std::string(filename);
        }
        return "";
#else
        // Console input fallback for Windows without dialogs
        std::cout << "\n=== File Selection ===" << std::endl;
        std::cout << "Enter input file path: ";
        std::string filename;
        std::getline(std::cin, filename);
        return filename;
#endif
#else
        // For Linux/Mac, use console input
        std::cout << "\n=== File Selection ===" << std::endl;
        std::cout << "Enter input file path: ";
        std::string filename;
        std::getline(std::cin, filename);
        return filename;
#endif
    }
    
    /**
     * @brief Show menu for file selection
     */
    std::string selectInputFile() {
        std::cout << "\n=== Input File Selection ===" << std::endl;
        std::cout << "1. Browse for file (open dialog)" << std::endl;
        std::cout << "2. Use simple_test_case.txt" << std::endl;
        std::cout << "3. Use enhanced_spe9_input.txt" << std::endl;
        std::cout << "4. Use spe10_benchmark.txt" << std::endl;
        std::cout << "5. Enter file path manually" << std::endl;
        std::cout << "6. Create new input file" << std::endl;
        std::cout << "7. Exit" << std::endl;
        std::cout << "\nEnter choice (1-7): ";
        
        int choice;
        std::cin >> choice;
        std::cin.ignore(); // Clear newline
        
        switch (choice) {
            case 1:
                return openFileDialog();
            case 2:
                return "simple_test_case.txt";
            case 3:
                return "enhanced_spe9_input.txt";
            case 4:
                benchmark_mode_ = true;
                return "spe10_benchmark.txt";
            case 5: {
                std::cout << "Enter file path: ";
                std::string path;
                std::getline(std::cin, path);
                return path;
            }
            case 6: {
                std::string filename = createNewInputFile();
                return filename;
            }
            case 7:
                return "";
            default:
                std::cout << "Invalid choice!" << std::endl;
                return "";
        }
    }
    
    /**
     * @brief Create a new input file interactively
     */
    std::string createNewInputFile() {
        std::cout << "\n=== Create New Input File ===" << std::endl;
        std::cout << "Enter filename (without extension): ";
        std::string basename;
        std::getline(std::cin, basename);
        std::string filename = basename + ".txt";
        
        std::cout << "\nSelect template:" << std::endl;
        std::cout << "1. Simple 2D waterflood" << std::endl;
        std::cout << "2. 3D five-spot pattern" << std::endl;
        std::cout << "3. Custom" << std::endl;
        std::cout << "Choice: ";
        
        int template_choice;
        std::cin >> template_choice;
        std::cin.ignore();
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot create file " << filename << std::endl;
            return "";
        }
        
        // Write header
        file << "# Enhanced Reservoir Simulation Input File\n";
        file << "# Generated by EnhancedReservoirSimulator v3.0\n";
        file << "# Date: " << getCurrentDateTime() << "\n\n";
        
        switch (template_choice) {
            case 1:
                writeSimple2DTemplate(file);
                break;
            case 2:
                write3DFiveSpotTemplate(file);
                break;
            case 3:
                writeCustomTemplate(file);
                break;
            default:
                writeSimple2DTemplate(file);
        }
        
        file.close();
        std::cout << "\nInput file created: " << filename << std::endl;
        return filename;
    }
    
    void writeSimple2DTemplate(std::ofstream& file) {
        file << "# Simple 2D Waterflood Template\n\n";
        
        file << "--GRID_NX\n20\n\n";
        file << "--GRID_NY\n20\n\n";
        file << "--GRID_NZ\n1\n\n";
        
        file << "--DOMAIN_LX\n1000.0\n\n";
        file << "--DOMAIN_LY\n1000.0\n\n";
        file << "--DOMAIN_LZ\n10.0\n\n";
        
        file << "--TIME_DT\n0.1\n\n";  // Smaller time step for stability
        file << "--TIME_TOTAL\n365.0\n\n";
        file << "--TIME_ADAPTIVE\nYES\n\n";
        file << "--TIME_DT_MAX\n5.0\n\n";
        file << "--TIME_DT_MIN\n0.001\n\n";
        file << "--TIME_CFL\n0.5\n\n";
        file << "--OUTPUT_FREQUENCY\n30\n\n";
        
        file << "--FLUID_RHO_OIL\n800.0\n\n";
        file << "--FLUID_RHO_WATER\n1000.0\n\n";
        file << "--FLUID_MU_OIL\n0.002\n\n";
        file << "--FLUID_MU_WATER\n0.001\n\n";
        file << "--FLUID_C_OIL\n1.0e-9\n\n";
        file << "--FLUID_C_WATER\n4.0e-10\n\n";
        file << "--ROCK_C_ROCK\n1.0e-10\n\n";
        
        file << "--ROCK_PERMX\n";
        for (int i = 0; i < 400; ++i) {
            file << "100.0 ";
            if ((i + 1) % 20 == 0) file << "\n";
        }
        file << "\n\n";
        
        file << "--ROCK_POROSITY\n";
        for (int i = 0; i < 400; ++i) {
            file << "0.2 ";
            if ((i + 1) % 20 == 0) file << "\n";
        }
        file << "\n\n";
        
        file << "--INIT_PRESSURE\n20000000.0\n\n";
        file << "--INIT_WATER_SAT\n0.2\n\n";
        
        file << "--RELPERM_MODEL\nCOREY\n\n";
        file << "--RELPERM_SWC\n0.2\n\n";
        file << "--RELPERM_SOR\n0.2\n\n";
        file << "--RELPERM_N_WATER\n2.0\n\n";
        file << "--RELPERM_N_OIL\n2.0\n\n";
        
        file << "--SOLVER_TYPE\nIMPES\n\n";
        file << "--SOLVER_P_TOL\n1.0e-6\n\n";
        file << "--SOLVER_MAX_P_ITER\n1000\n\n";
        file << "--SOLVER_OMEGA\n1.0\n\n";
        
        file << "--PHYS_MAX_DSAT\n0.05\n\n";
        
        file << "--WELLS\n";
        file << "# Name I J K_top K_bottom Type Control Target_value\n";
        file << "INJ1 1 1 1 1 INJECTOR RATE 50.0\n";  // Reduced rate
        file << "PROD1 20 20 1 1 PRODUCER BHP 10000000.0\n";
    }
    
    void write3DFiveSpotTemplate(std::ofstream& file) {
        file << "# 3D Five-Spot Pattern Template\n\n";
        
        file << "--GRID_NX\n30\n\n";
        file << "--GRID_NY\n30\n\n";
        file << "--GRID_NZ\n5\n\n";
        
        file << "--DOMAIN_LX\n1500.0\n\n";
        file << "--DOMAIN_LY\n1500.0\n\n";
        file << "--DOMAIN_LZ\n50.0\n\n";
        
        file << "--TIME_DT\n0.1\n\n";  // Small initial time step
        file << "--TIME_TOTAL\n1825.0  # 5 years\n\n";
        file << "--TIME_ADAPTIVE\nYES\n\n";
        file << "--TIME_DT_MAX\n10.0\n\n";
        file << "--TIME_DT_MIN\n0.001\n\n";
        file << "--TIME_CFL\n0.3\n\n";
        
        file << "--FLUID_RHO_OIL\n850.0\n\n";
        file << "--FLUID_RHO_WATER\n1020.0\n\n";
        file << "--FLUID_MU_OIL\n0.005\n\n";
        file << "--FLUID_MU_WATER\n0.0005\n\n";
        
        file << "--ROCK_PERMX\n";
        for (int k = 0; k < 5; ++k) {
            for (int j = 0; j < 30; ++j) {
                for (int i = 0; i < 30; ++i) {
                    double perm = 100.0 + 50.0 * sin(i * 0.2) * cos(j * 0.2);
                    file << perm << " ";
                }
                file << "\n";
            }
        }
        file << "\n\n";
        
        file << "--ROCK_POROSITY\n";
        for (int i = 0; i < 4500; ++i) {
            file << "0.25 ";
            if ((i + 1) % 30 == 0) file << "\n";
        }
        file << "\n\n";
        
        file << "--INIT_PRESSURE\n25000000.0\n\n";
        file << "--INIT_WATER_SAT\n0.15\n\n";
        
        file << "--RELPERM_MODEL\nCOREY\n\n";
        file << "--RELPERM_SWC\n0.15\n\n";
        file << "--RELPERM_SOR\n0.25\n\n";
        file << "--RELPERM_N_WATER\n2.5\n\n";
        file << "--RELPERM_N_OIL\n2.0\n\n";
        
        file << "--WELLS\n";
        file << "# Five-spot pattern\n";
        file << "INJ_CENTER 15 15 1 5 INJECTOR RATE 200.0\n";
        file << "PROD_NW 1 1 1 5 PRODUCER BHP 15000000.0\n";
        file << "PROD_NE 30 1 1 5 PRODUCER BHP 15000000.0\n";
        file << "PROD_SW 1 30 1 5 PRODUCER BHP 15000000.0\n";
        file << "PROD_SE 30 30 1 5 PRODUCER BHP 15000000.0\n";
        
        file << "\n--SOLVER_TYPE\nIMPES\n\n";
        file << "--SOLVER_OMEGA\n0.8\n\n";
        file << "--PHYS_MAX_DSAT\n0.02\n\n";
    }
    
    void writeCustomTemplate(std::ofstream& file) {
        // Interactive custom template creation
        std::cout << "\nEnter grid dimensions (NX NY NZ): ";
        int nx, ny, nz;
        std::cin >> nx >> ny >> nz;
        
        file << "--GRID_NX\n" << nx << "\n\n";
        file << "--GRID_NY\n" << ny << "\n\n";
        file << "--GRID_NZ\n" << nz << "\n\n";
        
        std::cout << "Enter domain size (LX LY LZ) in meters: ";
        double lx, ly, lz;
        std::cin >> lx >> ly >> lz;
        
        file << "--DOMAIN_LX\n" << lx << "\n\n";
        file << "--DOMAIN_LY\n" << ly << "\n\n";
        file << "--DOMAIN_LZ\n" << lz << "\n\n";
        
        // Add other necessary parameters with defaults
        file << "--TIME_DT\n0.1\n\n";
        file << "--TIME_TOTAL\n365.0\n\n";
        file << "--TIME_ADAPTIVE\nYES\n\n";
        file << "--OUTPUT_FREQUENCY\n30\n\n";
        
        // Default fluid properties
        file << "--FLUID_RHO_OIL\n800.0\n\n";
        file << "--FLUID_RHO_WATER\n1000.0\n\n";
        file << "--FLUID_MU_OIL\n0.002\n\n";
        file << "--FLUID_MU_WATER\n0.001\n\n";
        
        // Uniform permeability and porosity
        int n_cells = nx * ny * nz;
        
        file << "--ROCK_PERMX\n";
        for (int i = 0; i < n_cells; ++i) {
            file << "100.0 ";
            if ((i + 1) % 20 == 0) file << "\n";
        }
        file << "\n\n";
        
        file << "--ROCK_POROSITY\n";
        for (int i = 0; i < n_cells; ++i) {
            file << "0.2 ";
            if ((i + 1) % 20 == 0) file << "\n";
        }
        file << "\n\n";
        
        // Default initial conditions
        file << "--INIT_PRESSURE\n20000000.0\n\n";
        file << "--INIT_WATER_SAT\n0.2\n\n";
        
        // Default relative permeability
        file << "--RELPERM_MODEL\nCOREY\n\n";
        file << "--RELPERM_SWC\n0.2\n\n";
        file << "--RELPERM_SOR\n0.2\n\n";
        
        // Add one injector and one producer
        file << "--WELLS\n";
        file << "# Name I J K_top K_bottom Type Control Target_value\n";
        file << "INJ1 1 1 1 " << nz << " INJECTOR RATE 50.0\n";
        file << "PROD1 " << nx << " " << ny << " 1 " << nz << " PRODUCER BHP 10000000.0\n";
        
        std::cin.ignore(); // Clear buffer
    }
    
    std::string getCurrentDateTime() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::stringstream ss;
        #ifdef _MSC_VER
        // MSVC doesn't support std::put_time in older versions
        struct tm timeinfo;
        localtime_s(&timeinfo, &time_t);
        ss << std::setfill('0') 
           << timeinfo.tm_year + 1900 << "-"
           << std::setw(2) << timeinfo.tm_mon + 1 << "-"
           << std::setw(2) << timeinfo.tm_mday << " "
           << std::setw(2) << timeinfo.tm_hour << ":"
           << std::setw(2) << timeinfo.tm_min << ":"
           << std::setw(2) << timeinfo.tm_sec;
        #else
        struct tm* timeinfo = std::localtime(&time_t);
        if (timeinfo) {
            ss << std::setfill('0') 
               << timeinfo->tm_year + 1900 << "-"
               << std::setw(2) << timeinfo->tm_mon + 1 << "-"
               << std::setw(2) << timeinfo->tm_mday << " "
               << std::setw(2) << timeinfo->tm_hour << ":"
               << std::setw(2) << timeinfo->tm_min << ":"
               << std::setw(2) << timeinfo->tm_sec;
        } else {
            ss << "Unknown time";
        }
        #endif
        return ss.str();
    }
    
    /**
     * @brief Parse command line arguments
     */
    bool parseArguments(int argc, char* argv[]) {
        // If no arguments, show file dialog
        if (argc < 2) {
            input_file_ = selectInputFile();
            if (input_file_.empty()) {
                std::cout << "No file selected. Exiting." << std::endl;
                return false;
            }
            verbose_ = true; // Default to verbose when using dialog
            return true;
        }
        
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg == "-h" || arg == "--help") {
                showUsage(argv[0]);
                return false;
            } else if (arg == "-v" || arg == "--verbose") {
                verbose_ = true;
            } else if (arg == "-b" || arg == "--benchmark") {
                benchmark_mode_ = true;
            } else if (arg == "-o" || arg == "--output") {
                if (i + 1 < argc) {
                    output_dir_ = argv[++i];
                }
            } else if (arg == "-t" || arg == "--threads") {
                if (i + 1 < argc) {
                    int threads = std::stoi(argv[++i]);
                    #ifdef _OPENMP
                    omp_set_num_threads(threads);
                    #else
                    std::cout << "Warning: OpenMP not available, thread count ignored" << std::endl;
                    #endif
                }
            } else if (input_file_.empty()) {
                input_file_ = arg;
            }
        }
        
        if (input_file_.empty()) {
            std::cerr << "Error: No input file specified" << std::endl;
            showUsage(argv[0]);
            return false;
        }
        
        return true;
    }
    
    /**
     * @brief Show usage information
     */
    void showUsage(const char* program_name) {
        std::cout << "\nUsage: " << program_name << " [options] [input_file.txt]\n\n";
        std::cout << "If no input file is specified, a file dialog will open.\n\n";
        std::cout << "Options:\n";
        std::cout << "  -h, --help              Show this help message\n";
        std::cout << "  -v, --verbose           Enable verbose output\n";
        std::cout << "  -b, --benchmark         Run in benchmark mode\n";
        std::cout << "  -o, --output DIR        Set output directory\n";
        std::cout << "  -t, --threads N         Set number of OpenMP threads\n\n";
        std::cout << "Examples:\n";
        std::cout << "  " << program_name << "                    (opens file dialog)\n";
        std::cout << "  " << program_name << " spe9_input.txt\n";
        std::cout << "  " << program_name << " -v -o results/ enhanced_spe9.txt\n";
        std::cout << "  " << program_name << " -b -t 8 spe10_benchmark.txt\n\n";
    }
    
    /**
     * @brief Check if file exists
     */
    bool fileExists(const std::string& filename) {
        std::ifstream file(filename);
        return file.good();
    }
    
    /**
     * @brief Validate input file
     */
    bool validateInput() {
        if (!fileExists(input_file_)) {
            std::cerr << "Error: File not found: " << input_file_ << std::endl;
            return false;
        }
        
        std::ifstream file(input_file_);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open input file: " << input_file_ << std::endl;
            return false;
        }
        
        std::cout << "\n=== Input File Validation ===" << std::endl;
        std::cout << "Checking: " << input_file_ << std::endl;
        
        // Check for required keywords
        std::vector<std::string> required_keywords = {
            "GRID_NX", "GRID_NY", "GRID_NZ",
            "DOMAIN_LX", "DOMAIN_LY", "DOMAIN_LZ",
            "TIME_DT", "TIME_TOTAL",
            "FLUID_RHO_OIL", "FLUID_RHO_WATER",
            "FLUID_MU_OIL", "FLUID_MU_WATER",
            "ROCK_PERMX", "ROCK_POROSITY",
            "INIT_PRESSURE", "INIT_WATER_SAT",
            "WELLS"
        };
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        
        bool all_found = true;
        for (const auto& keyword : required_keywords) {
            if (content.find("--" + keyword) == std::string::npos) {
                std::cerr << "Missing required keyword: " << keyword << std::endl;
                all_found = false;
            }
        }
        
        if (all_found) {
            std::cout << "✓ All required keywords found" << std::endl;
            
            // Additional validation
            InputParser parser;
            if (parser.readFile(input_file_)) {
                int nx = parser.getInt("GRID_NX");
                int ny = parser.getInt("GRID_NY");
                int nz = parser.getInt("GRID_NZ");
                int expected_cells = nx * ny * nz;
                
                auto perm = parser.getDoubleVector("ROCK_PERMX");
                auto poro = parser.getDoubleVector("ROCK_POROSITY");
                
                if (perm.size() != expected_cells) {
                    std::cerr << "Warning: ROCK_PERMX size (" << perm.size() 
                             << ") doesn't match grid cells (" << expected_cells << ")" << std::endl;
                    all_found = false;
                }
                
                if (poro.size() != expected_cells) {
                    std::cerr << "Warning: ROCK_POROSITY size (" << poro.size() 
                             << ") doesn't match grid cells (" << expected_cells << ")" << std::endl;
                    all_found = false;
                }
                
                // Check physical validity
                for (double k : perm) {
                    if (k <= 0) {
                        std::cerr << "Error: Non-positive permeability found" << std::endl;
                        all_found = false;
                        break;
                    }
                }
                
                for (double phi : poro) {
                    if (phi <= 0 || phi > 1) {
                        std::cerr << "Error: Invalid porosity value found" << std::endl;
                        all_found = false;
                        break;
                    }
                }
            }
        }
        
        file.close();
        return all_found;
    }
    
    /**
     * @brief Run the simulation
     */
    bool runSimulation() {
        try {
            std::cout << "\n╔════════════════════════════════════════════════╗" << std::endl;
            std::cout <<   "║    ENHANCED RESERVOIR SIMULATOR v3.0           ║" << std::endl;
            std::cout <<   "║    World-Class Professional Edition            ║" << std::endl;
            std::cout <<   "║    Degined By Hamza Eldaw SUST                 ║" << std::endl;
            std::cout <<   "╚                                                ╝" << std::endl;
            std::cout <<   "╚════════════════════════════════════════════════╝" << std::endl;
            // Validate input
            if (!validateInput()) {
                return false;
            }
            
            // Set output directory
            if (!output_dir_.empty()) {
                #ifdef _WIN32
                _mkdir(output_dir_.c_str());
                #else
                mkdir(output_dir_.c_str(), 0755);
                #endif
            }
            
            // Start timing
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Create and run simulator
            std::cout << "\n=== Initializing Simulator ===" << std::endl;
            g_simulator = std::make_unique<EnhancedReservoirSimulator>(input_file_);
            
            std::cout << "\n=== Running Simulation ===" << std::endl;
            
            if (benchmark_mode_) {
                std::cout << "Running in BENCHMARK mode..." << std::endl;
            }
            
            g_simulator->run();
            
            // End timing
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
            
            // Generate summary report
            generateReport(duration.count());
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "\n❌ Simulation Error!" << std::endl;
            std::cerr << "Error: " << e.what() << std::endl;
            
            // Provide troubleshooting tips
            std::cout << "\nTroubleshooting tips:" << std::endl;
            std::cout << "1. Reduce initial time step (TIME_DT)" << std::endl;
            std::cout << "2. Enable adaptive time stepping" << std::endl;
            std::cout << "3. Reduce maximum saturation change (PHYS_MAX_DSAT)" << std::endl;
            std::cout << "4. Check well rates are reasonable" << std::endl;
            std::cout << "5. Verify fluid and rock properties" << std::endl;
            std::cout << "6. Use smaller CFL number" << std::endl;
            
            return false;
        }
    }
    
    /**
     * @brief Generate simulation summary report
     */
    void generateReport(long duration_seconds) {
        std::cout << "\n╔════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║         SIMULATION SUMMARY REPORT              ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════╝" << std::endl;
        
        std::cout << "\nInput file: " << input_file_ << std::endl;
        std::cout << "Total runtime: " << duration_seconds << " seconds";
        
        if (duration_seconds > 60) {
            std::cout << " (" << duration_seconds / 60 << " minutes " 
                     << duration_seconds % 60 << " seconds)";
        }
        std::cout << std::endl;
        
        if (duration_seconds > 0) {
            std::cout << "Performance rating: ";
            if (duration_seconds < 60) {
                std::cout << "⭐⭐⭐⭐⭐ Excellent (< 1 minute)" << std::endl;
            } else if (duration_seconds < 300) {
                std::cout << "⭐⭐⭐⭐ Good (< 5 minutes)" << std::endl;
            } else if (duration_seconds < 900) {
                std::cout << "⭐⭐⭐ Fair (< 15 minutes)" << std::endl;
            } else {
                std::cout << "⭐⭐ Consider optimization" << std::endl;
            }
        }
        
        std::cout << "\nOutput files generated:" << std::endl;
        std::cout << "  ✓ output_0.vtk (Initial conditions)" << std::endl;
        std::cout << "  ✓ output_*.vtk (Time steps)" << std::endl;
        std::cout << "  ✓ wells_*.csv (Well performance data)" << std::endl;
        std::cout << "  ✓ summary.csv (Field summary data)" << std::endl;
        
        std::cout << "\nVisualization instructions:" << std::endl;
        std::cout << "1. Open ParaView or your preferred visualization tool" << std::endl;
        std::cout << "2. Load all output_*.vtk files as a time series" << std::endl;
        std::cout << "3. Key variables to visualize:" << std::endl;
        std::cout << "   - water_saturation (range: 0.2 - 0.8)" << std::endl;
        std::cout << "   - pressure (in MPa)" << std::endl;
        std::cout << "   - oil_saturation" << std::endl;
        
        std::cout << "\nKey results to analyze:" << std::endl;
        std::cout << "  • Water breakthrough time at producers" << std::endl;
        std::cout << "  • Sweep efficiency evolution" << std::endl;
        std::cout << "  • Pressure maintenance" << std::endl;
        std::cout << "  • Recovery factor vs. time" << std::endl;
        std::cout << "  • Well productivity indices" << std::endl;
        
        if (benchmark_mode_) {
            std::cout << "\n📊 Benchmark Results:" << std::endl;
            std::cout << "  Cell updates per second: " 
                     << (g_simulator->getGrid().totalCells() * 
                        g_simulator->getSolver().getTimeStep()) / duration_seconds 
                     << std::endl;
        }
    }
};

/**
 * @brief Main entry point
 */
int main(int argc, char* argv[]) {
    // Install signal handlers
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // Set console encoding for better Unicode support
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    #endif
    
    auto manager = std::make_unique<SimulationManager>();
    
    try {
        // Show banner
        #ifdef _WIN32
        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        SetConsoleTextAttribute(hConsole, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
        #endif
        
        std::cout << "\n╔══════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                                                      ║" << std::endl;
        std::cout << "║      ENHANCED RESERVOIR SIMULATOR WITH IMPES         ║" << std::endl;
        std::cout << "║           World-Class Professional Edition           ║" << std::endl;
        std::cout << "║                    Version 3.0                       ║" << std::endl;
        std::cout << "║           Degined By Hamza Eldaw SUST                ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════╝" << std::endl;
        
        #ifdef _WIN32
        SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
        #endif
        
        // Parse command line arguments
        if (!manager->parseArguments(argc, argv)) {
            return 1;
        }
        
        // Run simulation
        bool success = manager->runSimulation();
        
        if (success) {
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
            #endif
            
            std::cout << "\n✅ Simulation completed successfully!" << std::endl;
            
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
            #endif
            
            std::cout << "\nNext steps:" << std::endl;
            std::cout << "1. Visualize results in ParaView" << std::endl;
            std::cout << "2. Run 'python Visualization.py' for GUI analysis" << std::endl;
            std::cout << "3. Check output files in the current directory" << std::endl;
            std::cout << "4. Compare results with analytical solutions" << std::endl;
        } else {
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_INTENSITY);
            #endif
            
            std::cout << "\n❌ Simulation failed!" << std::endl;
            
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
            #endif
        }
        
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        
        return success ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "\n💥 Critical error: " << e.what() << std::endl;
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        return 2;
    } catch (...) {
        std::cerr << "\n💥 Unknown error occurred!" << std::endl;
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        return 3;
    }
}