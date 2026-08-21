/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
 *
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file ReleaseType_continuous.hpp 
 * @brief This class represents a specific release type. 
 *
 * @note Child of ReleaseType
 * @sa ReleaseType
 */

#pragma once

#include "ReleaseType.hpp"

class ReleaseType_duration : public ReleaseType
{
private:
  // note that this also inherits data members ParticleReleaseType m_rType, int m_parPerTimestep, double m_releaseStartTime,
  //  double m_releaseEndTime, and int m_numPar from ReleaseType.
  // guidelines for how to set these variables within an inherited ReleaseType are given in ReleaseType.hpp.

  double releaseStartTime;
  double releaseEndTime;
  int parPerTimestep;


protected:
public:
  // Default constructor
  ReleaseType_duration()
  {
  }

  // destructor
  ~ReleaseType_duration()
  {
  }


  virtual void parseValues()
  {
    parReleaseType = ParticleReleaseType::duration;

    parsePrimitive<double>(true, releaseStartTime, "releaseStartTime");
    parsePrimitive<double>(true, releaseEndTime, "releaseEndTime");
    parsePrimitive<int>(true, parPerTimestep, "parPerTimestep");
  }


  void calcReleaseInfo(const double &timestep, const double &simDur)
{
    m_parPerTimestep = parPerTimestep;  // 设置初始粒子数
    m_releaseStartTime = releaseStartTime;
    m_releaseEndTime = releaseEndTime;
    double releaseDur = releaseEndTime - releaseStartTime;
    int nReleaseTimes = std::ceil(releaseDur / timestep);  // 计算总的释放次数

    m_numPar = 0;  // 初始化粒子总数为0

    // 遍历每个时间步，更新粒子数
    for (int i = 0; i < nReleaseTimes; ++i)
    {
        double currentTime = releaseStartTime + i * timestep;  // 当前时间

        // 如果当前时间在释放时间段内
        if (currentTime < releaseEndTime)
        {
            // 每秒增加一个粒子
            parPerTimestep = i + 1;  // 从第一个时间步开始，逐渐增加粒子数（每秒增加1个粒子）
        }

        // 累积粒子数
        m_numPar += parPerTimestep;
    }
}

};
