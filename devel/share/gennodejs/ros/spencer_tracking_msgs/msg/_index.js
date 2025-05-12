
"use strict";

let TrackedGroup = require('./TrackedGroup.js');
let TrackedPersons2d = require('./TrackedPersons2d.js');
let ImmDebugInfo = require('./ImmDebugInfo.js');
let CompositeDetectedPersons = require('./CompositeDetectedPersons.js');
let PersonTrajectoryEntry = require('./PersonTrajectoryEntry.js');
let TrackingTimingMetrics = require('./TrackingTimingMetrics.js');
let TrackedGroups = require('./TrackedGroups.js');
let ImmDebugInfos = require('./ImmDebugInfos.js');
let TrackedPerson = require('./TrackedPerson.js');
let DetectedPersons = require('./DetectedPersons.js');
let PersonTrajectory = require('./PersonTrajectory.js');
let CompositeDetectedPerson = require('./CompositeDetectedPerson.js');
let TrackedPersons = require('./TrackedPersons.js');
let DetectedPerson = require('./DetectedPerson.js');
let TrackedPerson2d = require('./TrackedPerson2d.js');

module.exports = {
  TrackedGroup: TrackedGroup,
  TrackedPersons2d: TrackedPersons2d,
  ImmDebugInfo: ImmDebugInfo,
  CompositeDetectedPersons: CompositeDetectedPersons,
  PersonTrajectoryEntry: PersonTrajectoryEntry,
  TrackingTimingMetrics: TrackingTimingMetrics,
  TrackedGroups: TrackedGroups,
  ImmDebugInfos: ImmDebugInfos,
  TrackedPerson: TrackedPerson,
  DetectedPersons: DetectedPersons,
  PersonTrajectory: PersonTrajectory,
  CompositeDetectedPerson: CompositeDetectedPerson,
  TrackedPersons: TrackedPersons,
  DetectedPerson: DetectedPerson,
  TrackedPerson2d: TrackedPerson2d,
};
