/**
 * Copyright 2013 Erik Zenker, Carlchristian Eckert, Marius Melzer
 *
 * This file is part of HASEonGPU
 *
 * HASEonGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HASEonGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HASEonGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


/**
 * @author Erik Zenker
 * @author Carlchristian Eckert
 * @licence GPLv3
 *
 */

#pragma once

#include <mesh.hpp>
#include <types.hpp>

/**
 * @brief A wrapper for calcPhiAse, that distributes sample points
 *        to the available MPI nodes.The Nodes will split 
 *        up in one head node and the others as compute nodes. 
 *        The head node distributes the available sample
 *        points by demand.
 *
 *
 * @return number of used compute nodes
 */
float calcPhiAseMPI (const ExperimentParameters &experiment,
		     const ComputeParameters &compute,
		     Mesh& mesh,
		     Result &result);

