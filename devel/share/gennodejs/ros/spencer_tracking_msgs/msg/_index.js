
"use strict";

let PersonTrajectory = require('./PersonTrajectory.js');
let TrackedPersons = require('./TrackedPersons.js');
let CompositeDetectedPersons = require('./CompositeDetectedPersons.js');
let DetectedPerson = require('./DetectedPerson.js');
let TrackedGroups = require('./TrackedGroups.js');
let CompositeDetectedPerson = require('./CompositeDetectedPerson.js');
let TrackedPersons2d = require('./TrackedPersons2d.js');
let PersonTrajectoryEntry = require('./PersonTrajectoryEntry.js');
let TrackingTimingMetrics = require('./TrackingTimingMetrics.js');
let TrackedGroup = require('./TrackedGroup.js');
let DetectedPersons = require('./DetectedPersons.js');
let ImmDebugInfos = require('./ImmDebugInfos.js');
let TrackedPerson = require('./TrackedPerson.js');
let ImmDebugInfo = require('./ImmDebugInfo.js');
let TrackedPerson2d = require('./TrackedPerson2d.js');

module.exports = {
  PersonTrajectory: PersonTrajectory,
  TrackedPersons: TrackedPersons,
  CompositeDetectedPersons: CompositeDetectedPersons,
  DetectedPerson: DetectedPerson,
  TrackedGroups: TrackedGroups,
  CompositeDetectedPerson: CompositeDetectedPerson,
  TrackedPersons2d: TrackedPersons2d,
  PersonTrajectoryEntry: PersonTrajectoryEntry,
  TrackingTimingMetrics: TrackingTimingMetrics,
  TrackedGroup: TrackedGroup,
  DetectedPersons: DetectedPersons,
  ImmDebugInfos: ImmDebugInfos,
  TrackedPerson: TrackedPerson,
  ImmDebugInfo: ImmDebugInfo,
  TrackedPerson2d: TrackedPerson2d,
};
