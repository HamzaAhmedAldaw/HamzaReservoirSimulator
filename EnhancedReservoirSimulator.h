#ifndef ENHANCED_RESERVOIR_SIMULATOR_H
#define ENHANCED_RESERVOIR_SIMULATOR_H

#include <vector>
#include <array>
#include <cmath>  // For std::sqrt, std::pow, std::isfinite, etc.
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>  // For std::min_element, std::max_element, std::minmax_element
#include <memory>
#include <string>
#include <iomanip>  // For std::setprecision, std::setw, etc.
#include <map>
#include <functional>
#include <stdexcept>
#include <numeric>  // For std::accumulate
#include <chrono>
#include <mutex>
#include <thread>
#include <atomic>
#include <cstring>  // For memcpy if needed
#include <iterator>  // For std::istreambuf_iterator

// OpenMP support if available
#ifdef _OPENMP
#include <omp.h>
#include <set>
#ifdef __has_include
    #if __has_include(<filesystem>)
        #include <filesystem>
    #endif
#endif
#endif

#ifdef _WIN32
#include <direct.h>  // For _mkdir
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// PERFORMANCE MONITOR
// ============================================================================

class PerformanceMonitor {
private:
    mutable std::mutex mutex_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::map<std::string, double> timings_;
    std::map<std::string, int> counts_;
    std::map<std::string, double> memory_usage_;
    
public:
    void startTimer(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    void endTimer(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>
                       (end_time - start_time_).count() / 1e6;
        
        timings_[name] += duration;
        counts_[name]++;
    }
    
    void recordMemory(const std::string& name, double mb) {
        std::lock_guard<std::mutex> lock(mutex_);
        memory_usage_[name] = mb;
    }
    
    void report() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "\n=== Performance Report ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        
        for (const auto& [name, time] : timings_) {
            std::cout << std::setw(30) << std::left << name << ": " 
                     << std::setw(10) << std::right << time << "s";
            if (counts_.count(name) > 0) {
                std::cout << " (" << counts_.at(name) << " calls, " 
                         << time/counts_.at(name) << "s avg)";
            }
            std::cout << std::endl;
        }
        
        if (!memory_usage_.empty()) {
            std::cout << "\n--- Memory Usage ---" << std::endl;
            for (const auto& [name, mb] : memory_usage_) {
                std::cout << std::setw(30) << std::left << name << ": " 
                         << std::setw(10) << std::right << mb << " MB" << std::endl;
            }
        }
    }
    
    double getTiming(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = timings_.find(name);
        return (it != timings_.end()) ? it->second : 0.0;
    }
};

// Global performance monitor
static PerformanceMonitor g_perfmon;

// ============================================================================
// SIMULATION CONFIGURATION
// ============================================================================

struct SimulationConfig {
    // Grid parameters
    int nx, ny, nz;
    double lx, ly, lz;
    std::vector<double> dx, dy, dz;
    
    // Time parameters
    double dt_initial;
    double dt_max;
    double dt_min;
    double total_time;
    int output_frequency;
    bool adaptive_timestep;
    double cfl_limit;
    double growth_factor;
    
    // Fluid properties
    double rho_oil, rho_water, rho_gas;
    double mu_oil, mu_water, mu_gas;
    double c_oil, c_water, c_gas;
    double c_rock;
    double p_ref;
    double bo_ref, bw_ref, bg_ref;
    
    // Rock properties
    std::vector<double> permeability_x;
    std::vector<double> permeability_y;
    std::vector<double> permeability_z;
    std::vector<double> porosity;
    
    // Initial conditions
    double initial_pressure;
    double initial_water_saturation;
    double initial_gas_saturation;
    double datum_depth;
    double woc_depth;
    double goc_depth;
    
    // Relative permeability
    std::string kr_model;
    double swc, sor, sgc, sorg;
    double n_water, n_oil, n_gas;
    double krw_max, kro_max, krg_max;
    bool use_hysteresis;
    
    // Capillary pressure
    std::string pc_model;
    double pc_entry_ow;
    
    // Well data
    struct WellData {
        std::string name;
        int i, j, k_top, k_bottom;
        std::string type;
        std::string control;
        double target_value;
        double radius;
        double skin;
        double min_bhp;
        double max_bhp;
    };
    std::vector<WellData> wells;
    
    // Solver parameters
    std::string solver_type;
    double pressure_tolerance;
    double saturation_tolerance;
    int max_pressure_iterations;
    int max_saturation_iterations;
    double omega;
    std::string preconditioner;
    
    // Physical constraints
    double min_saturation_change;
    double max_saturation_change;
    double min_pressure;
    double max_pressure;
    
    // Numerical options
    bool use_tvd;
    std::string flux_limiter;
    bool use_gravity;
    double gravity;
    
    // Output options
    bool output_vtk;
    bool output_restart;
    bool output_wells;
    bool output_summary;
    std::string output_dir;
    
    // Parallel options
    int num_threads;
    
    bool loadFromFile(const std::string& filename);
    void validate();
};

// ============================================================================
// INPUT PARSER
// ============================================================================

class InputParser {
private:
    std::map<std::string, std::vector<std::string>> data_;
    
public:
    bool readFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open input file " << filename << std::endl;
            return false;
        }
        
        std::string line;
        std::string current_keyword;
        std::vector<std::string> current_data;
        
        while (std::getline(file, line)) {
            // Remove comments
            size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }
            
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            
            if (line.empty()) continue;
            
            if (line.substr(0, 2) == "--") {
                if (!current_keyword.empty()) {
                    data_[current_keyword] = current_data;
                    current_data.clear();
                }
                current_keyword = line.substr(2);
                current_keyword.erase(0, current_keyword.find_first_not_of(" \t"));
                current_keyword.erase(current_keyword.find_last_not_of(" \t") + 1);
            } else {
                current_data.push_back(line);
            }
        }
        
        if (!current_keyword.empty()) {
            data_[current_keyword] = current_data;
        }
        
        file.close();
        return true;
    }
    
    double getDouble(const std::string& keyword, double default_value = 0.0) const {
        auto it = data_.find(keyword);
        if (it != data_.end() && !it->second.empty()) {
            try {
                return std::stod(it->second[0]);
            } catch (...) {
                std::cerr << "Warning: Invalid value for " << keyword << std::endl;
            }
        }
        return default_value;
    }
    
    int getInt(const std::string& keyword, int default_value = 0) const {
        auto it = data_.find(keyword);
        if (it != data_.end() && !it->second.empty()) {
            try {
                return std::stoi(it->second[0]);
            } catch (...) {
                std::cerr << "Warning: Invalid value for " << keyword << std::endl;
            }
        }
        return default_value;
    }
    
    std::string getString(const std::string& keyword, const std::string& default_value = "") const {
        auto it = data_.find(keyword);
        if (it != data_.end() && !it->second.empty()) {
            return it->second[0];
        }
        return default_value;
    }
    
    std::vector<double> getDoubleVector(const std::string& keyword) const {
        std::vector<double> result;
        auto it = data_.find(keyword);
        if (it != data_.end()) {
            for (const auto& line : it->second) {
                std::istringstream iss(line);
                double value;
                while (iss >> value) {
                    result.push_back(value);
                }
            }
        }
        return result;
    }
    
    bool hasKeyword(const std::string& keyword) const {
        return data_.find(keyword) != data_.end();
    }
    
    std::vector<std::string> getLines(const std::string& keyword) const {
        auto it = data_.find(keyword);
        if (it != data_.end()) {
            return it->second;
        }
        return std::vector<std::string>();
    }
};

// ============================================================================
// GRID SYSTEM
// ============================================================================

class Grid {
private:
    SimulationConfig& config_;
    std::vector<double> cell_volumes_;
    std::vector<double> cell_depths_;
    std::vector<double> transmissibility_x_;
    std::vector<double> transmissibility_y_;
    std::vector<double> transmissibility_z_;
    
public:
    Grid(SimulationConfig& config) : config_(config) {
        g_perfmon.startTimer("Grid initialization");
        calculateCellVolumes();
        calculateTransmissibilities();
        calculateCellDepths();
        g_perfmon.endTimer("Grid initialization");
    }
    
    void calculateCellVolumes() {
        cell_volumes_.resize(totalCells());
        
        #pragma omp parallel for collapse(3)
        for (int k = 0; k < config_.nz; ++k) {
            for (int j = 0; j < config_.ny; ++j) {
                for (int i = 0; i < config_.nx; ++i) {
                    int idx = index(i, j, k);
                    cell_volumes_[idx] = config_.dx[i] * config_.dy[j] * config_.dz[k];
                }
            }
        }
    }
    
    void calculateTransmissibilities() {
        int n_faces_x = (config_.nx - 1) * config_.ny * config_.nz;
        int n_faces_y = config_.nx * (config_.ny - 1) * config_.nz;
        int n_faces_z = config_.nx * config_.ny * (config_.nz - 1);
        
        transmissibility_x_.resize(n_faces_x, 0.0);
        transmissibility_y_.resize(n_faces_y, 0.0);
        transmissibility_z_.resize(n_faces_z, 0.0);
        
        // X-direction
        #pragma omp parallel for collapse(3)
        for (int k = 0; k < config_.nz; ++k) {
            for (int j = 0; j < config_.ny; ++j) {
                for (int i = 0; i < config_.nx - 1; ++i) {
                    int face_idx = i + j * (config_.nx - 1) + k * (config_.nx - 1) * config_.ny;
                    int idx1 = index(i, j, k);
                    int idx2 = index(i+1, j, k);
                    
                    double k1 = config_.permeability_x[idx1];
                    double k2 = config_.permeability_x[idx2];
                    double k_harm = 2.0 * k1 * k2 / (k1 + k2 + 1e-20);
                    
                    double area = config_.dy[j] * config_.dz[k];
                    double distance = 0.5 * (config_.dx[i] + config_.dx[i+1]);
                    
                    transmissibility_x_[face_idx] = k_harm * area / distance;
                }
            }
        }
        
        // Y-direction
        #pragma omp parallel for collapse(3)
        for (int k = 0; k < config_.nz; ++k) {
            for (int j = 0; j < config_.ny - 1; ++j) {
                for (int i = 0; i < config_.nx; ++i) {
                    int face_idx = i + j * config_.nx + k * config_.nx * (config_.ny - 1);
                    int idx1 = index(i, j, k);
                    int idx2 = index(i, j+1, k);
                    
                    double k1 = config_.permeability_y[idx1];
                    double k2 = config_.permeability_y[idx2];
                    double k_harm = 2.0 * k1 * k2 / (k1 + k2 + 1e-20);
                    
                    double area = config_.dx[i] * config_.dz[k];
                    double distance = 0.5 * (config_.dy[j] + config_.dy[j+1]);
                    
                    transmissibility_y_[face_idx] = k_harm * area / distance;
                }
            }
        }
        
        // Z-direction
        #pragma omp parallel for collapse(3)
        for (int k = 0; k < config_.nz - 1; ++k) {
            for (int j = 0; j < config_.ny; ++j) {
                for (int i = 0; i < config_.nx; ++i) {
                    int face_idx = i + j * config_.nx + k * config_.nx * config_.ny;
                    int idx1 = index(i, j, k);
                    int idx2 = index(i, j, k+1);
                    
                    double k1 = config_.permeability_z[idx1];
                    double k2 = config_.permeability_z[idx2];
                    double k_harm = 2.0 * k1 * k2 / (k1 + k2 + 1e-20);
                    
                    double area = config_.dx[i] * config_.dy[j];
                    double distance = 0.5 * (config_.dz[k] + config_.dz[k+1]);
                    
                    transmissibility_z_[face_idx] = k_harm * area / distance;
                }
            }
        }
    }
    
    void calculateCellDepths() {
        cell_depths_.resize(totalCells());
        
        for (int k = 0; k < config_.nz; ++k) {
            double z_top = 0.0;
            for (int kk = 0; kk < k; ++kk) {
                z_top += config_.dz[kk];
            }
            double z_center = z_top + 0.5 * config_.dz[k];
            
            for (int j = 0; j < config_.ny; ++j) {
                for (int i = 0; i < config_.nx; ++i) {
                    int idx = index(i, j, k);
                    cell_depths_[idx] = config_.datum_depth + z_center;
                }
            }
        }
    }
    
    int index(int i, int j, int k) const {
        return i + j * config_.nx + k * config_.nx * config_.ny;
    }
    
    void indices(int idx, int& i, int& j, int& k) const {
        k = idx / (config_.nx * config_.ny);
        int remainder = idx % (config_.nx * config_.ny);
        j = remainder / config_.nx;
        i = remainder % config_.nx;
    }
    
    int totalCells() const { 
        return config_.nx * config_.ny * config_.nz; 
    }
    
    double cellVolume(int idx) const {
        return cell_volumes_[idx];
    }
    
    double cellDepth(int idx) const {
        return cell_depths_[idx];
    }
    
    double transmissibilityX(int i, int j, int k) const {
        if (i >= config_.nx - 1) return 0.0;
        int face_idx = i + j * (config_.nx - 1) + k * (config_.nx - 1) * config_.ny;
        return transmissibility_x_[face_idx];
    }
    
    double transmissibilityY(int i, int j, int k) const {
        if (j >= config_.ny - 1) return 0.0;
        int face_idx = i + j * config_.nx + k * config_.nx * (config_.ny - 1);
        return transmissibility_y_[face_idx];
    }
    
    double transmissibilityZ(int i, int j, int k) const {
        if (k >= config_.nz - 1) return 0.0;
        int face_idx = i + j * config_.nx + k * config_.nx * config_.ny;
        return transmissibility_z_[face_idx];
    }
    
    std::vector<std::pair<int, double>> getNeighbors(int idx) const {
        std::vector<std::pair<int, double>> neighbors;
        int i, j, k;
        indices(idx, i, j, k);
        
        // X-direction neighbors
        if (i > 0) neighbors.push_back({index(i-1, j, k), transmissibilityX(i-1, j, k)});
        if (i < config_.nx - 1) neighbors.push_back({index(i+1, j, k), transmissibilityX(i, j, k)});
        
        // Y-direction neighbors
        if (j > 0) neighbors.push_back({index(i, j-1, k), transmissibilityY(i, j-1, k)});
        if (j < config_.ny - 1) neighbors.push_back({index(i, j+1, k), transmissibilityY(i, j, k)});
        
        // Z-direction neighbors
        if (k > 0) neighbors.push_back({index(i, j, k-1), transmissibilityZ(i, j, k-1)});
        if (k < config_.nz - 1) neighbors.push_back({index(i, j, k+1), transmissibilityZ(i, j, k)});
        
        return neighbors;
    }
};

// ============================================================================
// PROPERTY MODELS
// ============================================================================

class PropertyModels {
private:
    SimulationConfig& config_;
    
public:
    PropertyModels(SimulationConfig& config) : config_(config) {}
    
    double relativePermeabilityWater(double sw) const {
        sw = std::max(config_.swc, std::min(1.0 - config_.sor, sw));
        
        if (config_.kr_model == "COREY") {
            double sw_norm = (sw - config_.swc) / (1.0 - config_.swc - config_.sor);
            sw_norm = std::max(0.0, std::min(1.0, sw_norm));
            return config_.krw_max * std::pow(sw_norm, config_.n_water);
        }
        return 0.0;
    }
    
    double relativePermeabilityOil(double sw, double sg = 0.0) const {
        double so = 1.0 - sw - sg;
        so = std::max(config_.sor, so);
        
        if (config_.kr_model == "COREY") {
            double sw_norm = (sw - config_.swc) / (1.0 - config_.swc - config_.sor);
            sw_norm = std::max(0.0, std::min(1.0, sw_norm));
            return config_.kro_max * std::pow(1.0 - sw_norm, config_.n_oil);
        }
        return 0.0;
    }
    
    double capillaryPressure(double sw) const {
        if (config_.pc_model == "NONE") {
            return 0.0;
        }
        return 0.0; // Simplified
    }
    
    double waterMobility(double sw) const {
        return relativePermeabilityWater(sw) / config_.mu_water;
    }
    
    double oilMobility(double sw, double sg = 0.0) const {
        return relativePermeabilityOil(sw, sg) / config_.mu_oil;
    }
    
    double totalMobility(double sw, double sg = 0.0) const {
        return waterMobility(sw) + oilMobility(sw, sg);
    }
    
    double fractionalFlowWater(double sw, double sg = 0.0) const {
        double lambda_w = waterMobility(sw);
        double lambda_t = totalMobility(sw, sg);
        return (lambda_t > 0) ? lambda_w / lambda_t : 0.0;
    }
};

// ============================================================================
// WELL MODEL
// ============================================================================

class WellModel {
private:
    SimulationConfig& config_;
    Grid& grid_;
    PropertyModels& props_;
    
    struct WellPerformance {
        double cumulative_oil = 0.0;
        double cumulative_water = 0.0;
        double cumulative_gas = 0.0;
        double water_cut = 0.0;
        double last_rate_oil = 0.0;
        double last_rate_water = 0.0;
        double last_bhp = 0.0;
    };
    
    std::map<std::string, WellPerformance> performance_;
    
public:
    WellModel(SimulationConfig& config, Grid& grid, PropertyModels& props)
        : config_(config), grid_(grid), props_(props) {
        
        for (const auto& well : config_.wells) {
            performance_[well.name] = WellPerformance();
        }
    }
    
    double calculateWellIndex(const SimulationConfig::WellData& well, int k) const {
        int idx = grid_.index(well.i, well.j, k);
        double kx = config_.permeability_x[idx];
        double ky = config_.permeability_y[idx];
        
        double dx = config_.dx[well.i];
        double dy = config_.dy[well.j];
        double dz = config_.dz[k];
        
        // Peaceman model
        double r_eq = 0.28 * std::sqrt(dx * dx + dy * dy) / 2.0;
        double kh = std::sqrt(kx * ky);
        
        double WI = 2.0 * M_PI * kh * dz / (std::log(r_eq / well.radius) + well.skin);
        return WI;
    }
    
    void calculateProducerRates(const SimulationConfig::WellData& well,
                               const std::vector<double>& pressure,
                               const std::vector<double>& water_saturation,
                               double& q_oil, double& q_water,
                               double& bhp, double dt) {
        
        q_oil = q_water = 0.0;
        bhp = 0.0;
        double total_WI = 0.0;
        
        // Calculate average pressure
        for (int k = well.k_top; k <= well.k_bottom; ++k) {
            double WI = calculateWellIndex(well, k);
            int idx = grid_.index(well.i, well.j, k);
            bhp += WI * pressure[idx];
            total_WI += WI;
        }
        
        if (total_WI > 0) {
            bhp /= total_WI;
        }
        
        // Apply well control
        if (well.control == "BHP") {
            bhp = well.target_value;
        }
        
        // Calculate phase rates
        for (int k = well.k_top; k <= well.k_bottom; ++k) {
            double WI = calculateWellIndex(well, k);
            int idx = grid_.index(well.i, well.j, k);
            
            double dp = pressure[idx] - bhp;
            if (dp > 0) {
                double sw = water_saturation[idx];
                double lambda_o = props_.oilMobility(sw);
                double lambda_w = props_.waterMobility(sw);
                
                q_oil += WI * lambda_o * dp;
                q_water += WI * lambda_w * dp;
            }
        }
        
        // Update performance
        auto& perf = performance_[well.name];
        perf.water_cut = (q_oil + q_water > 0) ? q_water / (q_oil + q_water) : 0.0;
        perf.cumulative_oil += q_oil * dt;
        perf.cumulative_water += q_water * dt;
        perf.last_rate_oil = q_oil;
        perf.last_rate_water = q_water;
        perf.last_bhp = bhp;
    }
    
    double calculateInjectorRate(const SimulationConfig::WellData& well,
                                const std::vector<double>& pressure,
                                double& bhp) {
        
        double q_inj = 0.0;
        
        if (well.control == "RATE") {
            q_inj = well.target_value;
            
            // Calculate required BHP
            double total_mob_WI = 0.0;
            for (int k = well.k_top; k <= well.k_bottom; ++k) {
                double WI = calculateWellIndex(well, k);
                double lambda = props_.waterMobility(1.0 - config_.sor);
                total_mob_WI += WI * lambda;
            }
            
            if (total_mob_WI > 0) {
                // Average pressure in well cells
                double p_avg = 0.0;
                int count = 0;
                for (int k = well.k_top; k <= well.k_bottom; ++k) {
                    int idx = grid_.index(well.i, well.j, k);
                    p_avg += pressure[idx];
                    count++;
                }
                p_avg /= count;
                
                bhp = p_avg + q_inj / total_mob_WI;
            }
        } else { // BHP control
            bhp = well.target_value;
            
            // Calculate rate
            for (int k = well.k_top; k <= well.k_bottom; ++k) {
                double WI = calculateWellIndex(well, k);
                int idx = grid_.index(well.i, well.j, k);
                
                double dp = bhp - pressure[idx];
                if (dp > 0) {
                    double lambda = props_.waterMobility(1.0 - config_.sor);
                    q_inj += WI * lambda * dp;
                }
            }
        }
        
        return q_inj;
    }
    
    const WellPerformance& getPerformance(const std::string& well_name) const {
        static WellPerformance empty;
        auto it = performance_.find(well_name);
        return (it != performance_.end()) ? it->second : empty;
    }
    
    void updateCumulatives(const std::string& well_name, double dt) {
        auto& perf = performance_[well_name];
        // Cumulatives are already updated in calculateProducerRates
    }
};

// ============================================================================
// IMPES SOLVER (CORRECTED)
// ============================================================================

class IMPESSolver {
private:
    SimulationConfig& config_;
    Grid& grid_;
    PropertyModels& props_;
    WellModel& wells_;
    
    // Solution fields
    std::vector<double> pressure_;
    std::vector<double> water_saturation_;
    std::vector<double> gas_saturation_;
    std::vector<double> pressure_old_;
    std::vector<double> water_saturation_old_;
    
    // Intermediate fields
    std::vector<double> total_mobility_;
    std::vector<double> water_mobility_;
    std::vector<double> oil_mobility_;
    
    // Time tracking
    double current_time_;
    double current_dt_;
    int time_step_;
    
    // Performance tracking
    int pressure_iterations_;
    int total_pressure_iterations_;
    double cumulative_mass_error_;
    
public:
    IMPESSolver(SimulationConfig& config, Grid& grid, PropertyModels& props, WellModel& wells)
        : config_(config), grid_(grid), props_(props), wells_(wells),
          current_time_(0.0), current_dt_(config.dt_initial), time_step_(0),
          pressure_iterations_(0), total_pressure_iterations_(0),
          cumulative_mass_error_(0.0) {
        
        g_perfmon.startTimer("Solver initialization");
        
        int n_cells = grid_.totalCells();
        pressure_.resize(n_cells);
        water_saturation_.resize(n_cells);
        gas_saturation_.resize(n_cells, 0.0);
        
        // Initialize with equilibrium
        initializeEquilibrium();
        
        pressure_old_ = pressure_;
        water_saturation_old_ = water_saturation_;
        
        // Allocate intermediate fields
        total_mobility_.resize(n_cells);
        water_mobility_.resize(n_cells);
        oil_mobility_.resize(n_cells);
        
        updateMobilities();
        
        g_perfmon.endTimer("Solver initialization");
    }
    
    void initializeEquilibrium() {
        for (int idx = 0; idx < grid_.totalCells(); ++idx) {
            pressure_[idx] = config_.initial_pressure;
            water_saturation_[idx] = config_.initial_water_saturation;
            gas_saturation_[idx] = config_.initial_gas_saturation;
        }
    }
    
    void updateMobilities() {
        g_perfmon.startTimer("Mobility update");
        
        #pragma omp parallel for
        for (int i = 0; i < grid_.totalCells(); ++i) {
            water_mobility_[i] = props_.waterMobility(water_saturation_[i]);
            oil_mobility_[i] = props_.oilMobility(water_saturation_[i], gas_saturation_[i]);
            total_mobility_[i] = water_mobility_[i] + oil_mobility_[i];
        }
        
        g_perfmon.endTimer("Mobility update");
    }
    
    bool solvePressure() {
        g_perfmon.startTimer("Pressure solve");
        
        int n_cells = grid_.totalCells();
        std::vector<double> rhs(n_cells, 0.0);
        std::vector<double> diag(n_cells, 0.0);
        std::vector<std::vector<std::pair<int, double>>> off_diag(n_cells);
        
        // Build linear system
        buildPressureSystem(rhs, diag, off_diag);
        
        // Solve system
        bool converged = solvePreconditionedGaussSeidel(diag, off_diag, rhs, pressure_);
        
        total_pressure_iterations_ += pressure_iterations_;
        
        g_perfmon.endTimer("Pressure solve");
        
        if (converged) {
            std::cout << "  Pressure converged in " << pressure_iterations_ 
                     << " iterations" << std::endl;
        }
        
        return converged;
    }
    
    void buildPressureSystem(std::vector<double>& rhs,
                            std::vector<double>& diag,
                            std::vector<std::vector<std::pair<int, double>>>& off_diag) {
        
        int n_cells = grid_.totalCells();
        
        #pragma omp parallel for
        for (int idx = 0; idx < n_cells; ++idx) {
            diag[idx] = 0.0;
            rhs[idx] = 0.0;
            off_diag[idx].clear();
            
            // Get cell properties
            double phi = config_.porosity[idx];
            double vol = grid_.cellVolume(idx);
            
            // Compressibility
            double sw = water_saturation_[idx];
            double so = 1.0 - sw;
            double ct = config_.c_rock + sw * config_.c_water + so * config_.c_oil;
            
            // Accumulation term
            double accumulation = vol * phi * ct / current_dt_;
            diag[idx] += accumulation;
            rhs[idx] += accumulation * pressure_old_[idx];
            
            // Flow terms
            auto neighbors = grid_.getNeighbors(idx);
            
            for (const auto& [idx_nb, trans] : neighbors) {
                if (trans > 0) {
                    // Mobility at interface (upstream)
                    double mob_face = 0.5 * (total_mobility_[idx] + total_mobility_[idx_nb]);
                    
                    // Transmissibility term
                    double coeff = trans * mob_face;
                    
                    diag[idx] += coeff;
                    
                    #pragma omp critical
                    {
                        off_diag[idx].push_back({idx_nb, -coeff});
                    }
                }
            }
        }
        
        // Add well terms
        addWellTermsToPressureSystem(rhs, diag, off_diag);
    }
    
    void addWellTermsToPressureSystem(std::vector<double>& rhs,
                                     std::vector<double>& diag,
                                     std::vector<std::vector<std::pair<int, double>>>& off_diag) {
        
        for (const auto& well : config_.wells) {
            for (int k = well.k_top; k <= well.k_bottom; ++k) {
                int idx = grid_.index(well.i, well.j, k);
                double WI = wells_.calculateWellIndex(well, k);
                
                if (well.type == "PRODUCER") {
                    double mob = total_mobility_[idx];
                    double wc = WI * mob;
                    
                    if (well.control == "BHP") {
                        diag[idx] += wc;
                        rhs[idx] += wc * well.target_value;
                    }
                } else {  // INJECTOR
                    if (well.control == "RATE") {
                        double vol = grid_.cellVolume(idx);
                        double phi = config_.porosity[idx];
                        
                        // Distribute rate among completions
                        int n_compl = well.k_bottom - well.k_top + 1;
                        double q_cell = well.target_value / n_compl;
                        
                        // Source term (convert to reservoir volume rate)
                        double src_term = q_cell / (phi * vol / current_dt_);
                        
                        // Check for numerical issues
                        if (!std::isfinite(src_term)) {
                            std::cerr << "Warning: Non-finite source term at well " << well.name 
                                     << " cell (" << well.i << "," << well.j << "," << k << ")" << std::endl;
                            std::cerr << "  q_cell=" << q_cell << " phi=" << phi 
                                     << " vol=" << vol << " dt=" << current_dt_ << std::endl;
                            src_term = 0.0;
                        }
                        
                        rhs[idx] += src_term;
                    } else {  // BHP control
                        double mob = props_.waterMobility(1.0 - config_.sor);
                        double wc = WI * mob;
                        
                        diag[idx] += wc;
                        rhs[idx] += wc * well.target_value;
                    }
                }
            }
        }
    }
    
    bool solvePreconditionedGaussSeidel(const std::vector<double>& diag,
                                       const std::vector<std::vector<std::pair<int, double>>>& off_diag,
                                       const std::vector<double>& rhs,
                                       std::vector<double>& solution) {
        
        int n = diag.size();
        std::vector<double> solution_old = solution;
        double omega = config_.omega;
        double max_residual = 0.0;
        
        for (pressure_iterations_ = 0; 
             pressure_iterations_ < config_.max_pressure_iterations; 
             ++pressure_iterations_) {
            
            double max_change = 0.0;
            max_residual = 0.0;
            
            // Gauss-Seidel iteration
            for (int idx = 0; idx < n; ++idx) {
                double sum = rhs[idx];
                
                // Off-diagonal contributions
                for (const auto& [j, coeff] : off_diag[idx]) {
                    sum -= coeff * solution[j];
                }
                
                // Calculate residual
                double residual = sum - diag[idx] * solution[idx];
                max_residual = std::max(max_residual, std::abs(residual));
                
                // Update solution with relaxation
                if (std::abs(diag[idx]) > 1e-20) {
                    double new_value = sum / diag[idx];
                    solution[idx] = omega * new_value + (1.0 - omega) * solution[idx];
                    max_change = std::max(max_change, std::abs(solution[idx] - solution_old[idx]));
                }
            }
            
            // Check convergence
            if (max_change < config_.pressure_tolerance && 
                max_residual < config_.pressure_tolerance * config_.p_ref) {
                return true;
            }
            
            solution_old = solution;
        }
        
        // If we get here, solver didn't converge
        std::cerr << "Pressure solver: max iterations reached. Final residual: " 
                  << max_residual << ", tolerance: " << config_.pressure_tolerance * config_.p_ref << std::endl;
        
        // Return true if residual is reasonably small (relaxed convergence)
        return max_residual < config_.pressure_tolerance * config_.p_ref * 100;
    }
    
    bool solveSaturation() {
        g_perfmon.startTimer("Saturation solve");
        
        int n_cells = grid_.totalCells();
        std::vector<double> sw_new = water_saturation_;
        
        // Explicit saturation update
        #pragma omp parallel for
        for (int idx = 0; idx < n_cells; ++idx) {
            double phi = config_.porosity[idx];
            double vol = grid_.cellVolume(idx);
            double pv = phi * vol;
            
            double flux_w = 0.0;
            
            // Calculate fluxes from neighbors
            auto neighbors = grid_.getNeighbors(idx);
            
            for (const auto& [idx_nb, trans] : neighbors) {
                if (trans > 0) {
                    double dp = pressure_[idx] - pressure_[idx_nb];
                    
                    // Determine upstream
                    int idx_up = (dp >= 0) ? idx : idx_nb;
                    
                    // Calculate total flux
                    double mob_t = total_mobility_[idx_up];
                    double fw = props_.fractionalFlowWater(water_saturation_[idx_up]);
                    
                    double q_total = trans * mob_t * dp;
                    double q_water = q_total * fw;
                    
                    flux_w -= q_water;  // Negative because flux out is positive
                }
            }
            
            // Add well contributions
            int i, j, k;
            grid_.indices(idx, i, j, k);
            
            for (const auto& well : config_.wells) {
                if (well.i == i && well.j == j && k >= well.k_top && k <= well.k_bottom) {
                    if (well.type == "PRODUCER") {
                        double q_oil, q_water, bhp;
                        wells_.calculateProducerRates(well, pressure_, water_saturation_, 
                                                     q_oil, q_water, bhp, current_dt_);
                        
                        // Distribute among completions
                        int n_compl = well.k_bottom - well.k_top + 1;
                        flux_w -= q_water / n_compl;
                        
                    } else {  // INJECTOR
                        double bhp;
                        double q_inj = wells_.calculateInjectorRate(well, pressure_, bhp);
                        
                        int n_compl = well.k_bottom - well.k_top + 1;
                        flux_w += q_inj / n_compl;  // All water injection
                    }
                }
            }
            
            // Update saturation
            double dsw_dt = flux_w / pv;
            sw_new[idx] = water_saturation_[idx] + current_dt_ * dsw_dt;
            
            // Apply constraints
            sw_new[idx] = std::max(config_.swc, std::min(1.0 - config_.sor, sw_new[idx]));
        }
        
        // Update saturations
        water_saturation_ = sw_new;
        
        g_perfmon.endTimer("Saturation solve");
        
        return true;
    }
    
    void updateTimeStep() {
        if (!config_.adaptive_timestep) return;
        
        g_perfmon.startTimer("Time step calculation");
        
        // Calculate maximum saturation change
        double max_dsw = 0.0;
        
        #pragma omp parallel for reduction(max:max_dsw)
        for (int idx = 0; idx < grid_.totalCells(); ++idx) {
            double dsw = std::abs(water_saturation_[idx] - water_saturation_old_[idx]);
            max_dsw = std::max(max_dsw, dsw);
        }
        
        // Adjust time step
        if (max_dsw > config_.max_saturation_change) {
            current_dt_ *= 0.8;
        } else if (max_dsw < 0.5 * config_.max_saturation_change) {
            current_dt_ *= 1.2;
        }
        
        // Apply limits
        current_dt_ = std::max(config_.dt_min, std::min(config_.dt_max, current_dt_));
        
        // Don't exceed remaining time
        double remaining_time = config_.total_time - current_time_;
        current_dt_ = std::min(current_dt_, remaining_time);
        
        g_perfmon.endTimer("Time step calculation");
        
        std::cout << "  Next dt: " << current_dt_ / 86400.0 << " days" << std::endl;
    }
    
    double checkMassBalance() {
        // Calculate total mass in reservoir
        double total_mass_oil = 0.0;
        double total_mass_water = 0.0;
        double total_mass_oil_old = 0.0;
        double total_mass_water_old = 0.0;
        
        for (int i = 0; i < grid_.totalCells(); ++i) {
            double phi = config_.porosity[i];
            double vol = grid_.cellVolume(i);
            double pv = phi * vol;
            
            // Current masses
            double sw = water_saturation_[i];
            double so = 1.0 - sw;
            
            total_mass_water += pv * sw * config_.rho_water / config_.bw_ref;
            total_mass_oil += pv * so * config_.rho_oil / config_.bo_ref;
            
            // Old masses
            double sw_old = water_saturation_old_[i];
            double so_old = 1.0 - sw_old;
            
            total_mass_water_old += pv * sw_old * config_.rho_water / config_.bw_ref;
            total_mass_oil_old += pv * so_old * config_.rho_oil / config_.bo_ref;
        }
        
        // Calculate mass changes
        double mass_change_oil = total_mass_oil - total_mass_oil_old;
        double mass_change_water = total_mass_water - total_mass_water_old;
        
        // Calculate well contributions for this time step
        double well_mass_oil = 0.0;
        double well_mass_water = 0.0;
        
        for (const auto& well : config_.wells) {
            auto perf = wells_.getPerformance(well.name);
            
            if (well.type == "PRODUCER") {
                // Production removes mass (negative)
                well_mass_oil -= perf.last_rate_oil * current_dt_ * config_.rho_oil;
                well_mass_water -= perf.last_rate_water * current_dt_ * config_.rho_water;
            } else {
                // Injection adds mass (positive)
                double bhp;
                double q_inj = wells_.calculateInjectorRate(well, pressure_, bhp);
                well_mass_water += q_inj * current_dt_ * config_.rho_water;
            }
        }
        
        // Mass balance error
        double oil_error = std::abs(mass_change_oil - well_mass_oil);
        double water_error = std::abs(mass_change_water - well_mass_water);
        
        // Relative error (avoid division by zero)
        double total_mass = total_mass_oil + total_mass_water;
        double relative_error = (oil_error + water_error) / (total_mass + 1e-10);
        
        return relative_error;
    }
    
    void reportProgress() {
        double avg_pressure = std::accumulate(pressure_.begin(), pressure_.end(), 0.0) / 
                             pressure_.size() / 1e6;
        double avg_sw = std::accumulate(water_saturation_.begin(), 
                                       water_saturation_.end(), 0.0) / 
                       water_saturation_.size();
        
        std::cout << "Time: " << std::fixed << std::setprecision(1) 
                  << current_time_ / 86400.0 << " days, "
                  << "dt: " << current_dt_ / 86400.0 << " days, "
                  << "P_avg: " << avg_pressure << " MPa, "
                  << "Sw: " << std::setprecision(3) << avg_sw << std::endl;
        
        // Report well performance
        for (const auto& well : config_.wells) {
            auto perf = wells_.getPerformance(well.name);
            std::cout << "  " << well.name << ": ";
            
            if (well.type == "PRODUCER") {
                double rate_stb_d = perf.last_rate_oil * 86400.0 / config_.bo_ref;
                std::cout << "Oil: " << rate_stb_d << " STB/d, "
                         << "WCT: " << perf.water_cut * 100 << "%";
            } else {
                double rate_stb_d = perf.last_rate_water * 86400.0 / config_.bw_ref;
                std::cout << "Inj: " << rate_stb_d << " STB/d";
            }
            std::cout << std::endl;
        }
    }
    
    void writeOutput(int step) {
        g_perfmon.startTimer("Output writing");
        
        if (config_.output_vtk) {
            writeVTKOutput(step);
        }
        
        if (config_.output_wells) {
            writeWellReport(step);
        }
        
        if (config_.output_summary) {
            writeSummaryFile();
        }
        
        g_perfmon.endTimer("Output writing");
    }
    
    void writeVTKOutput(int step) {
        std::string filename = config_.output_dir + "/output_" + 
                              std::to_string(step) + ".vtk";
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open output file " << filename << std::endl;
            return;
        }
        
        // VTK header
        file << "# vtk DataFile Version 3.0\n";
        file << "Enhanced IMPES Simulation at time = " << current_time_/86400.0 << " days\n";
        file << "ASCII\n";
        file << "DATASET STRUCTURED_GRID\n";
        
        // Grid dimensions
        file << "DIMENSIONS " << config_.nx+1 << " " << config_.ny+1 << " " << config_.nz+1 << "\n";
        
        // Grid points
        int total_nodes = (config_.nx+1) * (config_.ny+1) * (config_.nz+1);
        file << "POINTS " << total_nodes << " float\n";
        
        for (int k = 0; k <= config_.nz; ++k) {
            double z = 0.0;
            for (int kk = 0; kk < k && kk < config_.nz; ++kk) {
                z += config_.dz[kk];
            }
            
            for (int j = 0; j <= config_.ny; ++j) {
                double y = 0.0;
                for (int jj = 0; jj < j && jj < config_.ny; ++jj) {
                    y += config_.dy[jj];
                }
                
                for (int i = 0; i <= config_.nx; ++i) {
                    double x = 0.0;
                    for (int ii = 0; ii < i && ii < config_.nx; ++ii) {
                        x += config_.dx[ii];
                    }
                    
                    file << x << " " << y << " " << z << "\n";
                }
            }
        }
        
        // Cell data
        file << "\nCELL_DATA " << grid_.totalCells() << "\n";
        
        // Pressure
        file << "\nSCALARS pressure float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (double p : pressure_) {
            file << p / 1e6 << "\n";  // Convert to MPa
        }
        
        // Water saturation
        file << "\nSCALARS water_saturation float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (double sw : water_saturation_) {
            file << sw << "\n";
        }
        
        // Oil saturation
        file << "\nSCALARS oil_saturation float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int i = 0; i < grid_.totalCells(); ++i) {
            file << (1.0 - water_saturation_[i] - gas_saturation_[i]) << "\n";
        }
        
        // Permeability
        file << "\nSCALARS permeability_x float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (double k : config_.permeability_x) {
            file << k * 1e15 << "\n";  // Convert to mD
        }
        
        // Porosity
        file << "\nSCALARS porosity float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (double phi : config_.porosity) {
            file << phi << "\n";
        }
        
        file.close();
        std::cout << "  Output written: " << filename << std::endl;
    }
    
    void writeWellReport(int step) {
        std::string filename = config_.output_dir + "/wells_" + 
                              std::to_string(step) + ".csv";
        
        std::ofstream file(filename);
        if (!file.is_open()) return;
        
        // Header
        file << "Well,Type,Time_days,Oil_rate_m3/d,Water_rate_m3/d,Gas_rate_m3/d,"
             << "BHP_MPa,Water_cut,GOR,Cumulative_oil_m3,Cumulative_water_m3,"
             << "Cumulative_gas_m3,Status\n";
        
        for (const auto& well : config_.wells) {
            auto perf = wells_.getPerformance(well.name);
            
            file << well.name << ","
                 << well.type << ","
                 << current_time_ / 86400.0 << ",";
            
            if (well.type == "PRODUCER") {
                file << perf.last_rate_oil * 86400.0 << ","
                     << perf.last_rate_water * 86400.0 << ","
                     << "0,"
                     << perf.last_bhp / 1e6 << ","
                     << perf.water_cut << ","
                     << "0,"
                     << perf.cumulative_oil << ","
                     << perf.cumulative_water << ","
                     << "0,"
                     << "ACTIVE\n";
            } else {
                file << "0,"
                     << perf.last_rate_water * 86400.0 << ","
                     << "0,"
                     << perf.last_bhp / 1e6 << ","
                     << "0,0,0,"
                     << perf.cumulative_water << ",0,"
                     << "ACTIVE\n";
            }
        }
        
        file.close();
    }
    
    void writeSummaryFile() {
        std::string filename = config_.output_dir + "/summary.csv";
        
        // Open in append mode
        std::ofstream file(filename, std::ios::app);
        if (!file.is_open()) return;
        
        // Write header if file is empty
        if (file.tellp() == 0) {
            file << "Time_days,Timestep,Avg_pressure_MPa,Avg_Sw,Avg_Sg,"
                 << "Total_oil_production_m3,Total_water_production_m3,"
                 << "Total_gas_production_m3,Total_water_injection_m3,"
                 << "Recovery_factor,Mass_balance_error\n";
        }
        
        // Calculate field averages
        double avg_p = std::accumulate(pressure_.begin(), pressure_.end(), 0.0) / 
                      pressure_.size() / 1e6;
        double avg_sw = std::accumulate(water_saturation_.begin(), 
                                       water_saturation_.end(), 0.0) / 
                       water_saturation_.size();
        double avg_sg = 0.0;
        
        // Calculate totals
        double total_oil_prod = 0.0, total_water_prod = 0.0;
        double total_water_inj = 0.0;
        
        for (const auto& well : config_.wells) {
            auto perf = wells_.getPerformance(well.name);
            if (well.type == "PRODUCER") {
                total_oil_prod += perf.cumulative_oil;
                total_water_prod += perf.cumulative_water;
            } else {
                total_water_inj += perf.cumulative_water;
            }
        }
        
        // Calculate recovery factor
        double ooip = calculateOOIP();
        double recovery_factor = (ooip > 0) ? total_oil_prod / ooip * 100.0 : 0.0;
        
        // Calculate mass balance error for this step
        double mass_error = checkMassBalance() * 100.0;  // Convert to percentage
        cumulative_mass_error_ += mass_error;
        
        // Write data
        file << current_time_ / 86400.0 << ","
             << time_step_ << ","
             << avg_p << ","
             << avg_sw << ","
             << avg_sg << ","
             << total_oil_prod << ","
             << total_water_prod << ","
             << "0,"
             << total_water_inj << ","
             << recovery_factor << ","
             << mass_error << "\n";
        
        file.close();
    }
    
    double calculateOOIP() {
        double ooip = 0.0;
        
        for (int i = 0; i < grid_.totalCells(); ++i) {
            double phi = config_.porosity[i];
            double vol = grid_.cellVolume(i);
            double so_initial = 1.0 - config_.initial_water_saturation;
            
            ooip += phi * vol * so_initial / config_.bo_ref;
        }
        
        return ooip;
    }
    
public:
    bool solve() {
        std::cout << "\n=== Starting Enhanced IMPES Simulation ===" << std::endl;
        std::cout << "Grid: " << config_.nx << "x" << config_.ny << "x" << config_.nz << std::endl;
        std::cout << "Total cells: " << grid_.totalCells() << std::endl;
        std::cout << "Initial time step: " << current_dt_ / 86400.0 << " days" << std::endl;
        
        try {
            // Write initial conditions
            writeOutput(0);
            
            // Main time loop
            while (current_time_ < config_.total_time) {
                time_step_++;
                
                g_perfmon.startTimer("Time step");
                
                // Store old solution
                pressure_old_ = pressure_;
                water_saturation_old_ = water_saturation_;
                
                // Update mobilities
                updateMobilities();
                
                // Solve pressure implicitly
                std::cout << "\nTime step " << time_step_ << " - Solving pressure..." << std::endl;
                
                if (!solvePressure()) {
                    std::cerr << "\n❌ ERROR: Pressure solution failed at time " 
                             << current_time_ / 86400.0 << " days" << std::endl;
                    std::cerr << "Possible causes:" << std::endl;
                    std::cerr << "- Time step too large (current: " << current_dt_/86400.0 << " days)" << std::endl;
                    std::cerr << "- Ill-conditioned matrix (check permeability contrast)" << std::endl;
                    std::cerr << "- Well rates too high" << std::endl;
                    
                    // Print pressure range for diagnostics
                    auto pmin_it = std::min_element(pressure_.begin(), pressure_.end());
                    auto pmax_it = std::max_element(pressure_.begin(), pressure_.end());
                    if (pmin_it != pressure_.end() && pmax_it != pressure_.end()) {
                        std::cerr << "Pressure range: " << *pmin_it/1e6 << " - " << *pmax_it/1e6 << " MPa" << std::endl;
                    }
                    
                    return false;
                }
                
                // Solve saturation explicitly
                std::cout << "Time step " << time_step_ << " - Solving saturation..." << std::endl;
                
                if (!solveSaturation()) {
                    std::cerr << "\n❌ ERROR: Saturation solution failed at time " 
                             << current_time_ / 86400.0 << " days" << std::endl;
                    std::cerr << "Possible causes:" << std::endl;
                    std::cerr << "- CFL violation" << std::endl;
                    std::cerr << "- Negative saturations" << std::endl;
                    
                    // Print saturation range for diagnostics
                    auto swmin_it = std::min_element(water_saturation_.begin(), water_saturation_.end());
                    auto swmax_it = std::max_element(water_saturation_.begin(), water_saturation_.end());
                    if (swmin_it != water_saturation_.end() && swmax_it != water_saturation_.end()) {
                        std::cerr << "Water saturation range: " << *swmin_it << " - " << *swmax_it << std::endl;
                    }
                    
                    return false;
                }
                
                // Update time
                current_time_ += current_dt_;
                
                // Check mass balance
                double mass_error = checkMassBalance();
                
                // Check for extreme mass balance error
                if (mass_error > 10.0) {  // 1000% error
                    std::cerr << "\n❌ ERROR: Extreme mass balance error: " << mass_error * 100.0 << "%" << std::endl;
                    std::cerr << "Simulation is unstable. Stopping." << std::endl;
                    return false;
                }
                
                // Update time step
                updateTimeStep();
                
                g_perfmon.endTimer("Time step");
                
                // Report progress
                if (time_step_ % 10 == 0) {
                    reportProgress();
                    std::cout << "  Mass balance error: " << mass_error * 100.0 << "%" << std::endl;
                }
                
                // Write output
                if (time_step_ % config_.output_frequency == 0) {
                    writeOutput(time_step_);
                }
                
                // Safety check - limit number of time steps to prevent infinite loops
                if (time_step_ > 10000) {
                    std::cerr << "\n⚠️ WARNING: Maximum time steps (10000) reached" << std::endl;
                    break;
                }
            }
            
            // Final output
            writeOutput(time_step_);
            
            std::cout << "\n=== Simulation Completed Successfully ===" << std::endl;
            std::cout << "Total time steps: " << time_step_ << std::endl;
            std::cout << "Average pressure iterations: " 
                      << total_pressure_iterations_ / time_step_ << std::endl;
            std::cout << "Average mass balance error: " 
                      << cumulative_mass_error_ / time_step_ << "%" << std::endl;
            
            // Performance report
            g_perfmon.report();
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "\n❌ EXCEPTION in solver: " << e.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << "\n❌ UNKNOWN EXCEPTION in solver" << std::endl;
            return false;
        }
    }
    
    // Getters
    const std::vector<double>& getPressure() const { return pressure_; }
    const std::vector<double>& getWaterSaturation() const { return water_saturation_; }
    double getCurrentTime() const { return current_time_; }
    double getTimeStep() const { return current_dt_; }
};

// ============================================================================
// MAIN SIMULATOR CLASS
// ============================================================================

class EnhancedReservoirSimulator {
private:
    SimulationConfig config_;
    std::unique_ptr<Grid> grid_;
    std::unique_ptr<PropertyModels> properties_;
    std::unique_ptr<WellModel> wells_;
    std::unique_ptr<IMPESSolver> solver_;
    
public:
    EnhancedReservoirSimulator(const std::string& input_file) {
        if (!config_.loadFromFile(input_file)) {
            throw std::runtime_error("Failed to load configuration file: " + input_file);
        }
        
        config_.validate();
        
        // Initialize components
        grid_ = std::make_unique<Grid>(config_);
        properties_ = std::make_unique<PropertyModels>(config_);
        wells_ = std::make_unique<WellModel>(config_, *grid_, *properties_);
        solver_ = std::make_unique<IMPESSolver>(config_, *grid_, *properties_, *wells_);
        
        std::cout << "\n=== Enhanced Reservoir Simulator Initialized ===" << std::endl;
        std::cout << "Grid: " << config_.nx << "x" << config_.ny << "x" << config_.nz 
                  << " (" << grid_->totalCells() << " cells)" << std::endl;
        std::cout << "Wells: " << config_.wells.size() << std::endl;
        
        // Memory usage
        double memory_mb = estimateMemoryUsage();
        std::cout << "Estimated memory usage: " << std::fixed << std::setprecision(1) 
                  << memory_mb << " MB" << std::endl;
        g_perfmon.recordMemory("Total", memory_mb);
    }
    
    void run() {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bool success = solver_->solve();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        std::cout << "\nTotal simulation time: " << duration.count() << " seconds" << std::endl;
        
        if (!success) {
            throw std::runtime_error("Simulation failed!");
        }
    }
    
    void stop() {
        // For async operation
    }
    
    const SimulationConfig& getConfig() const { return config_; }
    const Grid& getGrid() const { return *grid_; }
    const IMPESSolver& getSolver() const { return *solver_; }
    
    double estimateMemoryUsage() const {
        int n_cells = grid_->totalCells();
        
        // Solution arrays
        double solution_mb = 6 * n_cells * sizeof(double) / 1e6;
        
        // Grid and properties
        double grid_mb = 8 * n_cells * sizeof(double) / 1e6;
        
        return solution_mb + grid_mb;
    }
};

// Implementation of SimulationConfig methods
bool SimulationConfig::loadFromFile(const std::string& filename) {
    InputParser parser;
    if (!parser.readFile(filename)) {
        return false;
    }
    
    // Grid dimensions
    nx = parser.getInt("GRID_NX", 10);
    ny = parser.getInt("GRID_NY", 10);
    nz = parser.getInt("GRID_NZ", 5);
    
    // Domain size
    lx = parser.getDouble("DOMAIN_LX", 1000.0);
    ly = parser.getDouble("DOMAIN_LY", 1000.0);
    lz = parser.getDouble("DOMAIN_LZ", 10.0);
    
    // Grid spacing
    dx.resize(nx, lx / nx);
    dy.resize(ny, ly / ny);
    dz.resize(nz, lz / nz);
    
    // Time parameters
    dt_initial = parser.getDouble("TIME_DT", 1.0) * 86400.0;  // Convert days to seconds
    dt_max = parser.getDouble("TIME_DT_MAX", 10.0) * 86400.0;
    dt_min = parser.getDouble("TIME_DT_MIN", 0.01) * 86400.0;
    total_time = parser.getDouble("TIME_TOTAL", 365.0) * 86400.0;
    output_frequency = parser.getInt("OUTPUT_FREQUENCY", 30);
    adaptive_timestep = parser.getString("TIME_ADAPTIVE", "NO") == "YES";
    cfl_limit = parser.getDouble("TIME_CFL", 0.5);
    growth_factor = parser.getDouble("TIME_GROWTH_FACTOR", 1.2);
    
    // Fluid properties
    rho_oil = parser.getDouble("FLUID_RHO_OIL", 800.0);
    rho_water = parser.getDouble("FLUID_RHO_WATER", 1000.0);
    rho_gas = parser.getDouble("FLUID_RHO_GAS", 100.0);
    mu_oil = parser.getDouble("FLUID_MU_OIL", 1.0e-3);
    mu_water = parser.getDouble("FLUID_MU_WATER", 1.0e-3);
    mu_gas = parser.getDouble("FLUID_MU_GAS", 1.0e-5);
    c_oil = parser.getDouble("FLUID_C_OIL", 1.0e-9);
    c_water = parser.getDouble("FLUID_C_WATER", 4.0e-10);
    c_gas = parser.getDouble("FLUID_C_GAS", 1.0e-7);
    c_rock = parser.getDouble("ROCK_C_ROCK", 1.0e-10);
    p_ref = parser.getDouble("FLUID_P_REF", 1.0e7);
    bo_ref = parser.getDouble("FLUID_BO_REF", 1.0);
    bw_ref = parser.getDouble("FLUID_BW_REF", 1.0);
    bg_ref = parser.getDouble("FLUID_BG_REF", 0.01);
    
    // Rock properties
    if (parser.hasKeyword("ROCK_PERMX")) {
        permeability_x = parser.getDoubleVector("ROCK_PERMX");
        for (auto& k : permeability_x) k *= 1e-15;  // Convert mD to m²
    } else {
        permeability_x.resize(nx * ny * nz, 100e-15);
    }
    
    if (parser.hasKeyword("ROCK_PERMY")) {
        permeability_y = parser.getDoubleVector("ROCK_PERMY");
        for (auto& k : permeability_y) k *= 1e-15;
    } else {
        permeability_y = permeability_x;
    }
    
    if (parser.hasKeyword("ROCK_PERMZ")) {
        permeability_z = parser.getDoubleVector("ROCK_PERMZ");
        for (auto& k : permeability_z) k *= 1e-15;
    } else {
        permeability_z.resize(nx * ny * nz);
        for (size_t i = 0; i < permeability_z.size(); ++i) {
            permeability_z[i] = permeability_x[i] * 0.1;
        }
    }
    
    if (parser.hasKeyword("ROCK_POROSITY")) {
        porosity = parser.getDoubleVector("ROCK_POROSITY");
    } else {
        porosity.resize(nx * ny * nz, 0.2);
    }
    
    // Initial conditions
    initial_pressure = parser.getDouble("INIT_PRESSURE", 1.0e7);
    initial_water_saturation = parser.getDouble("INIT_WATER_SAT", 0.2);
    initial_gas_saturation = parser.getDouble("INIT_GAS_SAT", 0.0);
    datum_depth = parser.getDouble("INIT_DATUM_DEPTH", 0.0);
    woc_depth = parser.getDouble("INIT_WOC_DEPTH", 1000.0);
    goc_depth = parser.getDouble("INIT_GOC_DEPTH", 500.0);
    
    // Relative permeability
    kr_model = parser.getString("RELPERM_MODEL", "COREY");
    swc = parser.getDouble("RELPERM_SWC", 0.2);
    sor = parser.getDouble("RELPERM_SOR", 0.2);
    sgc = parser.getDouble("RELPERM_SGC", 0.0);
    sorg = parser.getDouble("RELPERM_SORG", 0.1);
    n_water = parser.getDouble("RELPERM_N_WATER", 2.0);
    n_oil = parser.getDouble("RELPERM_N_OIL", 2.0);
    n_gas = parser.getDouble("RELPERM_N_GAS", 2.0);
    krw_max = parser.getDouble("RELPERM_KRW_MAX", 1.0);
    kro_max = parser.getDouble("RELPERM_KRO_MAX", 1.0);
    krg_max = parser.getDouble("RELPERM_KRG_MAX", 1.0);
    use_hysteresis = parser.getString("RELPERM_HYSTERESIS", "NO") == "YES";
    
    // Capillary pressure
    pc_model = parser.getString("PCAP_MODEL", "NONE");
    pc_entry_ow = parser.getDouble("PCAP_ENTRY_OW", 5000.0);
    
    // Wells
    wells.clear();
    auto well_lines = parser.getLines("WELLS");
    for (const auto& line : well_lines) {
        std::istringstream iss(line);
        WellData well;
        
        iss >> well.name >> well.i >> well.j >> well.k_top >> well.k_bottom
            >> well.type >> well.control >> well.target_value;
        
        // Convert to 0-based indexing
        well.i--;
        well.j--;
        well.k_top--;
        well.k_bottom--;
        
        // Defaults
        well.radius = 0.1;
        well.skin = 0.0;
        well.min_bhp = 1.0e6;
        well.max_bhp = 100.0e6;
        
        // Convert rate from m³/day to m³/s
        if (well.control == "RATE") {
            well.target_value /= 86400.0;
        }
        
        wells.push_back(well);
    }
    
    // Solver parameters
    solver_type = parser.getString("SOLVER_TYPE", "IMPES");
    pressure_tolerance = parser.getDouble("SOLVER_P_TOL", 1.0e-6);
    saturation_tolerance = parser.getDouble("SOLVER_S_TOL", 1.0e-6);
    max_pressure_iterations = parser.getInt("SOLVER_MAX_P_ITER", 1000);
    max_saturation_iterations = parser.getInt("SOLVER_MAX_S_ITER", 100);
    omega = parser.getDouble("SOLVER_OMEGA", 1.0);
    preconditioner = parser.getString("SOLVER_PRECONDITIONER", "DIAGONAL");
    
    // Physical constraints
    min_saturation_change = parser.getDouble("PHYS_MIN_DSAT", 1.0e-6);
    max_saturation_change = parser.getDouble("PHYS_MAX_DSAT", 0.05);
    min_pressure = parser.getDouble("PHYS_MIN_PRESSURE", 1.0e5);
    max_pressure = parser.getDouble("PHYS_MAX_PRESSURE", 100.0e6);
    
    // Numerical options
    use_tvd = parser.getString("NUM_USE_TVD", "YES") == "YES";
    flux_limiter = parser.getString("NUM_FLUX_LIMITER", "VANLEER");
    use_gravity = parser.getString("NUM_USE_GRAVITY", "YES") == "YES";
    gravity = parser.getDouble("NUM_GRAVITY", 9.81);
    
    // Output options
    output_vtk = parser.getString("OUTPUT_VTK", "YES") == "YES";
    output_restart = parser.getString("OUTPUT_RESTART", "NO") == "YES";
    output_wells = parser.getString("OUTPUT_WELLS", "YES") == "YES";
    output_summary = parser.getString("OUTPUT_SUMMARY", "YES") == "YES";
    output_dir = parser.getString("OUTPUT_DIR", ".");
    
    // Parallel options
    num_threads = parser.getInt("PARALLEL_THREADS", 0);
    
    #ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    #endif
    
    return true;
}

void SimulationConfig::validate() {
    // Validate grid
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::runtime_error("Invalid grid dimensions");
    }
    
    // Validate array sizes
    if (dx.size() != nx) dx.resize(nx, lx / nx);
    if (dy.size() != ny) dy.resize(ny, ly / ny);
    if (dz.size() != nz) dz.resize(nz, lz / nz);
    
    int n_cells = nx * ny * nz;
    if (permeability_x.size() != n_cells) {
        throw std::runtime_error("Permeability X array size mismatch");
    }
    if (permeability_y.size() != n_cells) permeability_y = permeability_x;
    if (permeability_z.size() != n_cells) {
        permeability_z.resize(n_cells);
        for (int i = 0; i < n_cells; ++i) {
            permeability_z[i] = permeability_x[i] * 0.1;
        }
    }
    if (porosity.size() != n_cells) {
        throw std::runtime_error("Porosity array size mismatch");
    }
    
    // Validate fluid properties
    if (mu_oil <= 0 || mu_water <= 0) {
        throw std::runtime_error("Invalid fluid viscosity");
    }
    if (rho_oil <= 0 || rho_water <= 0) {
        throw std::runtime_error("Invalid fluid density");
    }
    
    // Validate saturations
    if (swc < 0 || swc > 1) throw std::runtime_error("Invalid Swc");
    if (sor < 0 || sor > 1) throw std::runtime_error("Invalid Sor");
    if (swc + sor > 1) throw std::runtime_error("Swc + Sor > 1");
    
    // Validate wells
    for (const auto& well : wells) {
        if (well.i < 0 || well.i >= nx || 
            well.j < 0 || well.j >= ny ||
            well.k_top < 0 || well.k_top >= nz ||
            well.k_bottom < well.k_top || well.k_bottom >= nz) {
            throw std::runtime_error("Invalid well location: " + well.name);
        }
        
        if (well.radius <= 0) {
            throw std::runtime_error("Invalid well radius: " + well.name);
        }
        
        if (well.type != "PRODUCER" && well.type != "INJECTOR") {
            throw std::runtime_error("Invalid well type: " + well.name);
        }
        
        if (well.control != "BHP" && well.control != "RATE") {
            throw std::runtime_error("Invalid well control: " + well.name);
        }
    }
    
    // Create output directory if needed
    if (!output_dir.empty() && output_dir != ".") {
        #ifdef _WIN32
        _mkdir(output_dir.c_str());
        #else
        mkdir(output_dir.c_str(), 0755);
        #endif
    }
}

#endif // ENHANCED_RESERVOIR_SIMULATOR_H